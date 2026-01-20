import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


## skeleton
# 我想我应该先抄写一遍minimal的代码

# x[Batch, Length, d_model]
# -> 输入投影
# x_proj[B,L,d_inner]
# -> SiLU,卷积,线性层:
# B[B,L,d_inner,d_state]
# C[B,L,d_state,d_inner]
# delta[B,L,dt_rank] # 没法直接用



# 模型参数，在加载模型时可选传入，否则使用默认
# 在开头的注释中有详细说明
@dataclass
class ModelArgs:
    d_model: int  # 模型输入的维度，由embedding决定
    n_layer: int  # mamba层数量
    vocab_size: int  # embedding层词表的大小
    d_state: int = 16  # 隐状态的维度，算法2中的N，每个序列只有一个隐状态，换句话说h[B,N]
    expand: int = 2  # SiLU激活前，线性输入扩展的比例，文中是2
    # rank of delta, 见3.6节，应该是说delta的维度，之后实际用到delta时再通过线性层投影到d_inner
    # 使得delta维度和投影后的d_inner相同，用于后续的离散化中（唯一用上delta的地方）
    dt_rank: Union[int, str] = 'auto'  # delta维度,默认为输入维度/16,
    d_conv: int = 4  # 卷积核 (仅出现一次在输入投影后)
    pad_vocab_size_multiple: int = 8  # 词表整数倍对齐, 或许只是个兼容性参数
    conv_bias: bool = True  # 卷积偏置
    bias: bool = True  # 输入输出投影中的偏置

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)  # 输入投影后,内部表示的维度

        if self.dt_rank == 'auto':
            # delta 的维度, 参与ssm计算前再线性投影到d_inner维
            # SiLU后的x[d_inner] -> 线性投影 -> delta[dt_rank] -> 线性投影 -> dt_proj[d_inner]
            # dt_rank较小,这是低秩投影的思想,减少参数量
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            # 词表整数倍对齐
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

class Mamba(nn.Module):
    def init(self, args: ModelArgs):
        super().__init__()
        pass

    def forward(self, input_ids):
        pass

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        pass

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        pass
    def forward(self, x):
        pass

class MambaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, input_ids):
        pass
    def ssm(self, x):
        pass
    def selective_scan(self, u, delta, A, B, C, D):
        pass

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        pass
    def forward(self,x):
        pass
