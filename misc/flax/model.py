import math
import json
from dataclasses import dataclass
from typing import Union, Optional, Any

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal, ones, zeros
import flax.linen as nn

@dataclass
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
        # 计算均方根
        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * weight

class MambaBlock(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, x):
        args = self.args
        b, l, d = x.shape

        # 1. 输入投影 (分支 1 和分支 2)
        x_and_res = nn.Dense(args.d_inner * 2, use_bias=args.bias, name='in_proj')(x)
        x_proj, res = jnp.split(x_and_res, 2, axis=-1)

        # 2. 一维卷积 (仅作用于分支 1)
        # 手动 padding (Causal padding: 左边补齐 kernel_size-1，右边不补)
        x_padded = jnp.pad(x_proj, ((0, 0), (args.d_conv - 1, 0), (0, 0)))
        x_conv = nn.Conv(features=args.d_inner,
                         kernel_size=(args.d_conv,),
                         feature_group_count=args.d_inner,
                         use_bias=args.conv_bias,
                         padding='VALID',
                         name='conv1d')(x_padded)

        # 3. 激活
        x_act = nn.silu(x_conv)
        res_act = nn.silu(res)

        # 4. 选择性 SSM
        y = self.s6(x_act)

        # 5. 合并残差分支
        y = y * res_act

        # 6. 输出投影
        output = nn.Dense(args.d_model, use_bias=args.bias, name='out_proj')(y)
        return output

    def s6(self, x):
        delta, A, B, C, D = self.generate_params(x)
        delta_A, delta_B = self.discretize_params(delta, A, B, use_zoh=False)
        return self.selective_scan(x, delta_A, delta_B, C, D)

    def generate_params(self, x):
        args = self.args

        # 初始化 A_log 和 D (移除 einops，使用 jnp 原生广播)
        n_vals = jnp.arange(1, args.d_state + 1, dtype=jnp.float32)
        A_init = jnp.broadcast_to(n_vals, (args.d_inner, args.d_state))

        A_log = self.param('A_log', lambda rng: jnp.log(A_init))
        D = self.param('D', ones, (args.d_inner,))

        A = -jnp.exp(A_log)

        # 动态参数
        dbc = nn.Dense(args.dt_rank + args.d_state * 2, use_bias=False, name='x_proj')(x)
        delta, B, C = jnp.split(dbc, [args.dt_rank, args.dt_rank + args.d_state], axis=-1)

        delta = nn.softplus(nn.Dense(args.d_inner, use_bias=True, name='dt_proj')(delta))

        return delta, A, B, C, D

    def discretize_params(self, delta, A, B, eps=1e-7, use_zoh=True):
        # A: (d_in, n), delta: (b, l, d_in), B: (b, l, n)
        A_exp = jnp.einsum('bld,dn->bldn', delta, A)
        delta_A = jnp.exp(A_exp)

        if use_zoh:
            A_b = jnp.expand_dims(A, axis=(0, 1))
            zoh_factor = (delta_A - 1.0) / (A_b + eps)
            delta_B = jnp.einsum('bldn,bln->bldn', zoh_factor, B)
        else:
            delta_B = jnp.einsum('bld,bln->bldn', delta, B)

        return delta_A, delta_B

    def selective_scan(self, u, delta_A, delta_B, C, D):
        b, l, d_in, n = delta_A.shape
        delta_B_u = delta_B * jnp.expand_dims(u, -1)

        # 使用 jax.lax.scan 替代 python for 循环，获得极致性能
        def scan_fn(carry, inputs):
            dA, dBu, C_i = inputs
            carry = dA * carry + dBu
            y_i = jnp.einsum('bdn,bn->bd', carry, C_i)
            return carry, y_i

        # 准备 scan 输入: 将时间维度 l 放到第 0 维
        # inputs shape: (l, b, d_in, n) ...
        scan_inputs = (
            jnp.swapaxes(delta_A, 0, 1),
            jnp.swapaxes(delta_B_u, 0, 1),
            jnp.swapaxes(C, 0, 1)
        )
        init_carry = jnp.zeros((b, d_in, n), dtype=delta_A.dtype)

        _, ys = jax.lax.scan(scan_fn, init_carry, scan_inputs)

        # 将输出从 (l, b, d_in) 变回 (b, l, d_in)
        y = jnp.swapaxes(ys, 0, 1)
        y = y + u * D
        return y

class ResidualBlock(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, x):
        normed = RMSNorm(self.args.d_model, name='norm')(x)
        output = MambaBlock(self.args, name='mixer')(normed) + x
        return output

class Mamba(nn.Module):
    args: ModelArgs

    @nn.compact
    def __call__(self, input_ids):
        # Embedding
        x = nn.Embed(num_embeddings=self.args.vocab_size,
                     features=self.args.d_model,
                     name='embedding')(input_ids)

        for i in range(self.args.n_layer):
            x = ResidualBlock(self.args, name=f'layers_{i}')(x)

        x = RMSNorm(self.args.d_model, name='norm_f')(x)

        # 权重共享 lm_head
        # 在 Flax 中，如果要严格绑定词向量权重，最简单的方法是直接矩阵乘法
        embedding_variables = self.variables.get('params', {}).get('embedding')
        if embedding_variables is not None and 'embedding' in embedding_variables:
            # 推理阶段或已初始化时直接使用 embed 权重转置
            emb_weight = embedding_variables['embedding']
            logits = jnp.dot(x, emb_weight.T)
        else:
            # 初始化占位或 fallback (严格对齐可使用 nn.Dense 参数绑定)
            logits = nn.Dense(self.args.vocab_size, use_bias=False, name='lm_head')(x)

        return logits