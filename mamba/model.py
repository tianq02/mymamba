import math
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dataclasses import dataclass

from einops import repeat, einsum


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
    """
    完整的mamba模型，包含：
    - 首层的embedding
    - 中间连续多个ResidualBlock(mamba模型+包裹的残差连接和归一化)
    - 最后的f范数
    """
    def init(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # 1. embedding 层，将输入id映射到嵌入向量
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        # 2. Mamba模块们，这里写ResidualBlock，对应IBM文档插图中最右侧的残差链接和底下的Norm
        # 代码中的实现好像和原文有点区别
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        # 3. 在输出前，通过F范数（平方和的平方根）归一化
        self.norm_f = RMSNorm(args.d_model)
        # 4. lmhead线性层，转换隐藏表示回到词表id，
        # 这里直接拿embedding的权重来用，就是逆向的embedding
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        pass

    def forward(self, input_ids):
        """
        :param input_ids: 词表id序列, shape(Batch, Length)
        :return: 每个token的后续预测概率, shape(Batch, Length, vocab_size)
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    # 这里直接偷了
    #
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))


        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model

# 残差块,把mamba包装加上残差链接和归一化
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # mixer: 被混合的mamba模块
        self.mixer = MambaBlock(args)
        # RMSNorm: 不带中心的LayerNorm
        self.norm = RMSNorm(args.d_model)
    def forward(self, x):
        """

        :param x: shape(b,l,d_model)每个token的'嵌入'
        :return: shape(b,l,d_model)
        """
        # mamba minimal 中对于残差块的实现有些讨论,这里我们按照原文写了一个
        # 如果炸了,换成这个 output = self.mixer(self.norm(x)) + x
        output = self.norm(self.mixer(x)+x)
        return output

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # 输入映射, * 2是指一半进ssm,一半到右边仅激活（mamba内类似残差的分支，不是前面残差块在mamba外套的残差）,两个线性计算合并
        # 有关d_inner,这是模块内的输入扩展,见ModelArgs.expand,默认参数下,d_inner=d_model*2
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # 1维卷积,仅用于ssm分支,在silu激活前(激活没有参数,不在这里写)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # 生成动态参数:
        # 原文是拿激活后的数据通过一个线性层得到delta,B,C
        # 然后delta再经过一次"Broadcast"层,扩展维度(两步放一起就是低秩投影)再激活
        # 这里是SiLU到SSM中间的小线性层,同上,合并参数
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        # delta专属的Broadcast线性层,扩展维度回到d_inner
        self.dt_proj = nn.Linear(args.d_inner, args.d_inner)

        # 偷来的,初始化A的参数为[1,2,3,4; 1,2,3,4; 1,2,3,4]
        # 这可不是什么HiPPO
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        # A中每项取自然对数
        self.A_log = nn.Parameter(torch.log(A))
        # 初始化D为全1的向量
        self.D = nn.Parameter(torch.ones(args.d_inner))

        # 输出投影，再把中间维度投影回到词表的维度
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """
        直接求解mamba的输入输出：
        :param x: shape(b,l,d_model) 上一层的输出/embedding
        :return: shape(b,l,d_model)
        """

        # mamba_minimal中d用的是d_inner,但我觉得应该是d_model
        # 我感觉不太对,还没做输入投影哪来的inner,应该是注释标错了
        [b,l,d] = x.shape

        # 计算扩展x和残差分支
        x_and_res = self.in_proj(x) # (b,l,2*d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner,self.args.d_inner], dim=-1)

        # 我想我对卷积的理解可能有点问题，这里的卷积似乎是逐个特征维度分别做的
        # update: 不是，就n_in入通道n_in出通道，n*n个一维的卷积核
        # 而且，如果marteen的文章有参考价值的话，mamba中卷积的padding方式对最终性能有影响
        # 这里有一点点不明确，目前按照mamba_minimal的方式处理
        # 1. 交换x的维度
        x = x.rearrange(x,'b l d_in -> b d_in l') # 把l挪到最后一维，在l上一维卷积
        # 2. 一维卷积(从后往前，前面的维度视作入通道，后面的维度视作下标)
        # 切片，避免padding造成的维度变化，这里最后切一刀，相当于丢掉输入末尾的padding，保持输出维度一致
        # 理论上应该也可以用F.pad实现，真要折腾的话及的改self.conv1d定义中的padding
        x = self.conv1d(x)[:,:,:self.args.d_inner]
        # 3. 再转回去
        x = x.rearrange(x,'b d_in,l -> b l d_in')

        # SiLU激活
        x = F.silu(x)
        res = F.silu(res)

        # 选择性ssm, 包含delta,B,C的计算
        # 三个步骤：计算delta,B,C; 转换成离散参数；循环计算SSM
        y = self.s6(x)

        # 合并残差分支
        y = y * res

        # 输出投影
        output = self.out_proj(y)

        return output

    def s6(self, x):
        (delta,A,B,C,D) = self.generate_params(x)  # 生成时变参数
        (AA,BB) = self.discretize_params(delta,A,B)  # ZOH离散化A,B
        y = self.selective_scan(x,AA,BB,C,D)  # 选择性扫描
        return y

    def generate_params(self,x):
        """
        生成时变的SSM参数
        :param x: shape(b, l, d_in), mamba块中的序列中间表示
        :return: tuple[delta(b,l,dt_rank),A(d_in,n),B(b,l,n),C(b,l,n),D(d_in,)]
        """
        (d_in,n) = self.A_log.shape

        # AD时不变
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        # deltaBC时变,由x经过线性变换产生
        dbc = self.x_proj(x)  # (b,l,dt_rank+2n), 连一起的delta,B,C
        (delta,B,C) = dbc.split(split_size=[self.args.dt_rank,n,n], dim=-1)

        # delta经过第二个线性层广播,再激活
        delta = F.softplus(self.dt_proj(delta))

        # 注意这里返回的都是连续形式
        return delta,A,B,C,D

    @staticmethod
    def discretize_params(delta, A, B, eps: float = 1e-7, use_zoh: bool = True):
        """
        ZOH离散化参数A,B
        :param delta: 持续时间
        :param A: (d_in,n)状态更新函数, 非时变
        :param B: (b,l,n)输入函数, 时变(体现为bl)
        :param eps: ZOH离散化中, 避免除0错误的分母添加量
        :param use_zoh: 参数B的离散化方式, True时ZOH, False时Euler(更快)
        :return:
            delta_A: (b, l, d_in, n)   -- 离散化A = exp(delta * A)
            delta_B: (b, l, d_in, n)   -- 离散化B（ZOH 或 Euler）
        """
        # A: (d_in, n), delta: (b, l, d_in), B: (b, l, n)
        # delta_A = exp(delta * A)  -> (b, l, d_in, n)
        A_exp = einsum(delta, A, 'b l d_in, d_in n -> b l d_in n')
        delta_A = torch.exp(A_exp)

        if use_zoh:
            # ZOH: ((exp(A*dt) - 1) / A) * B
            A_b = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_in, n) for broadcasting
            zoh_factor = (delta_A - 1.0) / (A_b + eps)  # (b, l, d_in, n)
            # delta_B shape (b, l, d_in, n)
            delta_B = einsum(zoh_factor, B, 'b l d_in n, b l n -> b l d_in n')
        else:
            # 简单 Euler 离散化
            delta_B = einsum(delta, B, 'b l d_in, b l n -> b l d_in n')

        return delta_A, delta_B

    def selective_scan(self, u, delta_A, delta_B, C, D)->torch.Tensor:
        """
        选择性扫描算法
        :param u: (b, l, d_in)mamba内部表示x改名字
        :param delta_A: (b, l, d_in, n), 离散化A
        :param delta_B: (b, l, d_in, n), 离散化B
        :param C: (b, l, n)
        :param D: (d_in)
        :return: (b, l, d_in)
        """
        (b, l, d_in, n) = delta_A.shape

        # 把 delta_B 与 u 相乘得到最终的 deltaB_u: (b, l, d_in, n)
        delta_B_u = delta_B * u.unsqueeze(-1)  # 广播 (b, l, d_in, 1) -> (b, l, d_in, n)

        x = torch.zeros((b, d_in, n), device=delta_A.device)
        ys = []
        for i in range(l):
            x = delta_A[:, i] * x + delta_B_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output