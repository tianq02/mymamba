import math
from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp
from jax.nn.initializers import ones
import flax.linen as nn

@dataclass(unsafe_hash=True)
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    use_zoh: bool = False
    # 注意：离散化模式取决于模型，而官方模型的训练时用的是欧拉离散化
    # use_zoh应当永远设为False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', ones, (self.d_model,))
        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * weight

class MambaBlock(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, x):
        """处理整个序列 (Prefill)"""
        args = self.args
        b, l, d = x.shape

        x_and_res = nn.Dense(args.d_inner * 2, use_bias=args.bias, name='in_proj')(x)
        x_proj, res = jnp.split(x_and_res, 2, axis=-1)

        x_padded = jnp.pad(x_proj, ((0, 0), (args.d_conv - 1, 0), (0, 0)))

        # 提取最后的卷积状态，供下一步 step 使用
        final_conv_state = x_padded[:, -(args.d_conv - 1):, :]

        x_conv = nn.Conv(features=args.d_inner, kernel_size=(args.d_conv,),
                         feature_group_count=args.d_inner, use_bias=args.conv_bias,
                         padding='VALID', name='conv1d')(x_padded)

        x_act = nn.silu(x_conv)
        res_act = nn.silu(res)

        y, final_ssm_state = self.s6(x_act)
        y = y * res_act

        output = nn.Dense(args.d_model, use_bias=args.bias, name='out_proj')(y)
        return output, (final_conv_state, final_ssm_state)


    @nn.compact
    def step(self, x, conv_state, ssm_state):
        """处理单步序列 (Decoding)"""
        args = self.args
        x_and_res = nn.Dense(args.d_inner * 2, use_bias=args.bias, name='in_proj')(x)
        x_proj, res = jnp.split(x_and_res, 2, axis=-1)

        # 1. 卷积状态更新
        conv_inputs = jnp.concatenate([conv_state, x_proj[:, None, :]], axis=1)
        new_conv_state = conv_inputs[:, 1:, :]

        x_conv = nn.Conv(features=args.d_inner, kernel_size=(args.d_conv,),
                         feature_group_count=args.d_inner, use_bias=args.conv_bias,
                         padding='VALID', name='conv1d')(conv_inputs)
        x_conv = jnp.squeeze(x_conv, axis=1)

        x_act = nn.silu(x_conv)
        res_act = nn.silu(res)

        # 2. SSM 状态更新
        # generate_params 内部的 nn.Dense 默认作用于最后一个维度，完美支持 (b, d_inner)
        delta, A, B, C, D = self.generate_params(x_act)

        # discretize_params 经过广播逻辑改造后，同样兼容 2D 输入
        DA, DB = self.discretize_params(delta, A, B, use_zoh=self.args.use_zoh)

        # 状态转移: new_ssm = DA * ssm + DB * x
        # 此时 DA 和 DB 的 shape 为 (b, d_inner, d_state)，x_act 需要在最后加一个维度对齐
        new_ssm_state = DA * ssm_state + DB * x_act[:, :, None]

        # 计算输出
        y = jnp.sum(new_ssm_state * C[:, None, :], axis=-1)
        y = y + x_act * D
        y = y * res_act

        output = nn.Dense(args.d_model, use_bias=args.bias, name='out_proj')(y)
        return output, new_conv_state, new_ssm_state

    def s6(self, x):
        delta, A, B, C, D = self.generate_params(x)
        # 修复：必须使用 use_zoh=False，和原版 PyTorch 保持一致
        DA, DB = self.discretize_params(delta, A, B, use_zoh=self.args.use_zoh)
        return self.selective_scan(x, DA, DB, C, D)

    def generate_params(self, x):
        args = self.args
        n_vals = jnp.arange(1, args.d_state + 1, dtype=jnp.float32)
        A_init = jnp.broadcast_to(n_vals, (args.d_inner, args.d_state))
        A_log = self.param('A_log', lambda rng: jnp.log(A_init))
        D = self.param('D', ones, (args.d_inner,))
        A = -jnp.exp(A_log)

        dbc = nn.Dense(args.dt_rank + args.d_state * 2, use_bias=False, name='x_proj')(x)
        delta, B, C = jnp.split(dbc, [args.dt_rank, args.dt_rank + args.d_state], axis=-1)
        delta = nn.softplus(nn.Dense(args.d_inner, use_bias=True, name='dt_proj')(delta))
        return delta, A, B, C, D

    def discretize_params(self, delta, A, B, eps=1e-7, use_zoh=False):
        # 1. 计算 \overline{A} = \exp(\Delta A)  % ZOH
        delta_A = delta[..., None] * A  # (b, l, d, n) 或 (b, d, n)
        DA = jnp.exp(delta_A)

        # 2. 对齐 B 的维度
        # Prefill 阶段 (b, l, d) -> (b, l, 1, n)
        # Decoding 阶段 (b, d) -> (b, 1, n)
        B_expanded = jnp.expand_dims(B, axis=-2)

        # 3. 计算 \overline{B}
        if use_zoh:
            # \overline{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
            #              = (DA - I) / A \cdot B
            DB = (DA - 1.0) / A * B_expanded
        else:
            # \overline{B} = \Delta B  % Euler
            DB = delta[..., None] * B_expanded

        return DA, DB

    def selective_scan(self, u, DA, DB, C, D):
        b, l, d_in, n = DA.shape
        DB_u = DB * jnp.expand_dims(u, -1)

        def scan_fn(carry, inputs):
            dA, dBu, C_i = inputs
            carry = dA * carry + dBu
            y_i = jnp.einsum('bdn,bn->bd', carry, C_i)
            return carry, y_i

        scan_inputs = (
            jnp.swapaxes(DA, 0, 1), jnp.swapaxes(DB_u, 0, 1), jnp.swapaxes(C, 0, 1)
        )
        init_carry = jnp.zeros((b, d_in, n), dtype=DA.dtype)

        # carry 就是跑完序列后最终的 ssm_state
        carry, ys = jax.lax.scan(scan_fn, init_carry, scan_inputs)

        y = jnp.swapaxes(ys, 0, 1)
        y = y + u * D
        return y, carry

class ResidualBlock(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, x):
        normed = RMSNorm(self.args.d_model, name='norm')(x)
        output, states = MambaBlock(self.args, name='mixer')(normed)
        return output + x, states

    @nn.compact
    def step(self, x, conv_state, ssm_state):
        normed = RMSNorm(self.args.d_model, name='norm')(x)
        output, new_conv_state, new_ssm_state = MambaBlock(self.args, name='mixer').step(normed, conv_state, ssm_state)
        return output + x, new_conv_state, new_ssm_state

class Mamba(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, input_ids):
        x = nn.Embed(num_embeddings=self.args.vocab_size, features=self.args.d_model, name='embedding')(input_ids)
        states = []
        for i in range(self.args.n_layer):
            x, layer_states = ResidualBlock(self.args, name=f'layers_{i}')(x)
            states.append(layer_states)

        x = RMSNorm(self.args.d_model, name='norm_f')(x)
        embedding_variables = self.variables.get('params', {}).get('embedding')
        emb_weight = embedding_variables['embedding']
        logits = jnp.dot(x, emb_weight.T)
        return logits, states

    @nn.compact
    def step(self, input_id, states):
        # 注意这里 input_id 是 (batch,) 形状，仅一个 token
        x = nn.Embed(num_embeddings=self.args.vocab_size, features=self.args.d_model, name='embedding')(input_id)
        new_states = []
        for i in range(self.args.n_layer):
            conv_state, ssm_state = states[i]
            x, new_conv, new_ssm = ResidualBlock(self.args, name=f'layers_{i}').step(x, conv_state, ssm_state)
            new_states.append((new_conv, new_ssm))

        x = RMSNorm(self.args.d_model, name='norm_f')(x)
        embedding_variables = self.variables.get('params', {}).get('embedding')
        emb_weight = embedding_variables['embedding']
        logits = jnp.dot(x, emb_weight.T)
        return logits, new_states