from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Callable, Optional, List
from jax.nn.initializers import ones, constant, uniform
import math

# 注意：JAX 的默认图像维度顺序是 (B, H, W, C)，这与 PyTorch 的 (B, C, H, W) 不同。
# 下面的实现假设输入是通道在后的格式 (B, H, W, C)。

class Downsample(nn.Module):
    dim: int
    keep_dim: bool = False

    @nn.compact
    def __call__(self, x):
        dim_out = self.dim if self.keep_dim else 2 * self.dim
        x = nn.Conv(features=dim_out, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False)(x)
        return x

class PatchEmbed(nn.Module):
    in_dim: int = 64
    dim: int = 96

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Conv(features=self.in_dim, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-4)(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.dim, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-4)(x)
        x = nn.relu(x)
        return x

class ConvBlock(nn.Module):
    dim: int
    kernel_size: int = 3
    layer_scale: Optional[float] = None
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = False):
        input_x = x
        x = nn.Conv(features=self.dim, kernel_size=(self.kernel_size, self.kernel_size), strides=(1, 1), padding=((1, 1), (1, 1)))(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        x = nn.gelu(x, approximate=True)

        x = nn.Conv(features=self.dim, kernel_size=(self.kernel_size, self.kernel_size), strides=(1, 1), padding=((1, 1), (1, 1)))(x)
        x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)

        if self.layer_scale is not None:
            gamma = self.param('gamma', nn.initializers.constant(self.layer_scale), (self.dim,))
            x = x * gamma

        # 简单模拟 DropPath (Flax 原生不带 DropPath，需要手动借助 nn.Dropout 改造，这里简写直接相加)
        return input_x + x

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = False):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads

        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = head_dim ** -0.5
        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        if train and self.attn_drop > 0:
            attn_weights = nn.Dropout(rate=self.attn_drop)(attn_weights, deterministic=False)

        x_out = jnp.matmul(attn_weights, v)
        x_out = jnp.transpose(x_out, (0, 2, 1, 3))
        x_out = jnp.reshape(x_out, (B, N, self.dim))

        x_out = nn.Dense(self.dim)(x_out)
        if train and self.proj_drop > 0:
            x_out = nn.Dropout(rate=self.proj_drop)(x_out, deterministic=False)
        return x_out

class MambaVisionMixer(nn.Module):
    d_model: int
    d_state: int = 16
    d_conv: int = 3  # MambaVision 中默认是 3
    expand: int = 1  # MambaVision 中 transformer_blocks 以外的 block 默认 expand=1

    def setup(self):
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        # 定义 A_log 参数
        n_vals = jnp.arange(1, self.d_state + 1, dtype=jnp.float32)
        A_init = jnp.broadcast_to(n_vals, (self.d_inner // 2, self.d_state))
        self.A_log = self.param('A_log', lambda rng: jnp.log(A_init))
        self.D = self.param('D', ones, (self.d_inner // 2,))

    @nn.compact
    def __call__(self, hidden_states, train: bool = False):
        B, L, D = hidden_states.shape
        d_half = self.d_inner // 2

        # 1. Input Projection
        xz = nn.Dense(self.d_inner, use_bias=False, name='in_proj')(hidden_states)
        x, z = jnp.split(xz, 2, axis=-1)

        # 2. 1D Convolution (对 x 和 z 分别卷积)
        # 在 Flax 中处理序列数据时，L 为空间维，特征维在后
        x = nn.Conv(features=d_half, kernel_size=(self.d_conv,),
                    feature_group_count=d_half, use_bias=False,
                    padding='SAME', name='conv1d_x')(x)
        x = nn.silu(x)

        z = nn.Conv(features=d_half, kernel_size=(self.d_conv,),
                    feature_group_count=d_half, use_bias=False,
                    padding='SAME', name='conv1d_z')(z)
        z = nn.silu(z)

        # 3. 提取 SSM 参数
        # x_dbl 映射得到 dt, B, C
        x_dbl = nn.Dense(self.dt_rank + self.d_state * 2, use_bias=False, name='x_proj')(x)
        dt, B_ssm, C_ssm = jnp.split(x_dbl, [self.dt_rank, self.dt_rank + self.d_state], axis=-1)

        # dt_proj (相当于 Mamba1 中的 dt_proj，这里由于 softplus=True，需加上 softplus 激活)
        dt = nn.Dense(d_half, use_bias=True, name='dt_proj')(dt)
        dt = nn.softplus(dt)

        # 4. 离散化与扫描计算 (复用提供的 SSM 逻辑)
        A = -jnp.exp(self.A_log)

        # 离散化 (Euler 法)
        A_exp = jnp.einsum('bld,dn->bldn', dt, A)
        delta_A = jnp.exp(A_exp)
        delta_B = jnp.einsum('bld,bln->bldn', dt, B_ssm)

        # SSM 扫描序列 (jax.lax.scan)
        y = self._selective_scan(x, delta_A, delta_B, C_ssm, self.D)

        # 5. Output Projection (注意 MambaVision 是 cat拼接 而不是相乘)
        y = jnp.concatenate([y, z], axis=-1)
        out = nn.Dense(self.d_model, use_bias=False, name='out_proj')(y)

        return out

    def _selective_scan(self, u, delta_A, delta_B, C, D):
        """复用自 Mamba-1 的 jax.lax.scan SSM 计算"""
        b, l, d_in, n = delta_A.shape
        delta_B_u = delta_B * jnp.expand_dims(u, -1)

        def scan_fn(carry, inputs):
            dA, dBu, C_i = inputs
            carry = dA * carry + dBu  # (B, D, N)
            y_i = jnp.einsum('bdn,bn->bd', carry, C_i)
            return carry, y_i

        scan_inputs = (
            jnp.swapaxes(delta_A, 0, 1),
            jnp.swapaxes(delta_B_u, 0, 1),
            jnp.swapaxes(C, 0, 1)
        )
        init_carry = jnp.zeros((b, d_in, n), dtype=delta_A.dtype)

        # 执行序列扫描
        _, ys = jax.lax.scan(scan_fn, init_carry, scan_inputs)

        y = jnp.swapaxes(ys, 0, 1)
        y = y + u * D
        return y

class Mlp(nn.Module):
    hidden_features: int
    out_features: Optional[int] = None
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = False):
        out_features = self.out_features or x.shape[-1]
        x = nn.Dense(self.hidden_features)(x)
        x = nn.gelu(x)
        if train and self.drop > 0:
            x = nn.Dropout(rate=self.drop)(x, deterministic=False)
        x = nn.Dense(out_features)(x)
        if train and self.drop > 0:
            x = nn.Dropout(rate=self.drop)(x, deterministic=False)
        return x

class Block(nn.Module):
    dim: int
    num_heads: int
    counter: int
    transformer_blocks: List[int]
    mlp_ratio: float = 4.0
    layer_scale: Optional[float] = None

    @nn.compact
    def __call__(self, x, train: bool = False):
        norm1 = nn.LayerNorm()(x)

        if self.counter in self.transformer_blocks:
            mixer_out = Attention(dim=self.dim, num_heads=self.num_heads)(norm1, train=train)
        else:
            # mixer_out = MambaVisionMixer(d_model=self.dim)(norm1, train=train)
            mixer_out = MambaVisionMixer(d_model=self.dim, d_state=8)(norm1, train=train)


        if self.layer_scale is not None:
            gamma_1 = self.param('gamma_1', nn.initializers.constant(self.layer_scale), (self.dim,))
            mixer_out = mixer_out * gamma_1

        x = x + mixer_out

        norm2 = nn.LayerNorm()(x)
        mlp_out = Mlp(hidden_features=int(self.dim * self.mlp_ratio))(norm2, train=train)

        if self.layer_scale is not None:
            gamma_2 = self.param('gamma_2', nn.initializers.constant(self.layer_scale), (self.dim,))
            mlp_out = mlp_out * gamma_2

        x = x + mlp_out
        return x

class MambaVisionLayer(nn.Module):
    dim: int
    depth: int
    num_heads: int
    window_size: int
    conv: bool = False
    downsample: bool = True
    mlp_ratio: float = 4.0
    layer_scale: Optional[float] = None
    transformer_blocks: List[int] = ()

    @nn.compact
    def __call__(self, x, train: bool = False):
        # 假设输入为 (B, H, W, C)
        B, H, W, C = x.shape

        if self.conv:
            for i in range(self.depth):
                x = ConvBlock(dim=self.dim, layer_scale=self.layer_scale)(x, train=train)
        else:
            # 窗口分区逻辑 (Window Partition)
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))

            Hp, Wp = x.shape[1], x.shape[2]

            # 转换为窗口
            x = jnp.reshape(x, (B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C))
            x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
            x = jnp.reshape(x, (-1, self.window_size * self.window_size, C))

            # 执行 Blocks
            for i in range(self.depth):
                x = Block(dim=self.dim, num_heads=self.num_heads, counter=i,
                          transformer_blocks=self.transformer_blocks,
                          mlp_ratio=self.mlp_ratio, layer_scale=self.layer_scale)(x, train=train)

            # 窗口反向恢复
            num_windows = (Hp // self.window_size) * (Wp // self.window_size)
            x = jnp.reshape(x, (B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1))
            x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
            x = jnp.reshape(x, (B, Hp, Wp, C))

            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]

        if self.downsample:
            x = Downsample(dim=self.dim)(x)

        return x

class MambaVision(nn.Module):
    dim: int = 80
    in_dim: int = 32
    depths: Sequence[int] = (1, 3, 8, 4)
    window_size: Sequence[int] = (8, 8, 14, 7)
    mlp_ratio: float = 4.0
    num_heads: Sequence[int] = (2, 4, 8, 16)
    num_classes: int = 1000
    layer_scale: Optional[float] = None

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = PatchEmbed(in_dim=self.in_dim, dim=self.dim)(x, train=train)

        for i in range(len(self.depths)):
            is_conv = True if i in [0, 1] else False
            downsample = (i < 3)
            # 根据原始逻辑，奇数和偶数 depth 有不同的 transformer_blocks 切分
            d = self.depths[i]
            t_blocks = list(range(d//2 + 1, d)) if d % 2 != 0 else list(range(d//2, d))

            x = MambaVisionLayer(
                dim=int(self.dim * (2 ** i)),
                depth=self.depths[i],
                num_heads=self.num_heads[i],
                window_size=self.window_size[i],
                conv=is_conv,
                downsample=downsample,
                mlp_ratio=self.mlp_ratio,
                layer_scale=self.layer_scale,
                transformer_blocks=t_blocks
            )(x, train=train)

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = jnp.mean(x, axis=(1, 2)) # Global Average Pooling

        if self.num_classes > 0:
            x = nn.Dense(self.num_classes)(x)

        return x

# 实例化示例 (MambaVision-T)
# model = MambaVision(dim=80, in_dim=32, depths=[1, 3, 8, 4], num_heads=[2, 4, 8, 16], window_size=[8, 8, 14, 7])
# params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)), train=False)