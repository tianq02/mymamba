# pytorch speedrun

1. Install PyTorch

    听我的，用docker环境预装的，别没事找事

2. 很抱歉我现在屁都不会，接下来我们逐个示例程序看

## 示例程序们：

### 1. numpy example

在torch之前，直接使用numpy搓机器学习是怎么样的？

[code](./1.py)

```python
# -*- coding: utf-8 -*-
import numpy as np
import math


# 随机输入输出数据，后面用于训练
# 训练中将使用这些数据拟合一个三次多项式
# 具体而言，用3次多项式去拟合sin函数在[-π, π]区间内的值
# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 训练前，初始化三次多项式的系数（随机）
# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

# 学习率，决定每次更新的幅度
learning_rate = 1e-6
for t in range(2000):
    # 前向传播：计算预测的y，用于计算损失
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 计算并打印损失，这里是平方误差
    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # 反向传播，计算a, b, c, d相对于loss的梯度
    # 说明：
    # 1. grad_y_pred是loss对y_pred的梯度，因为loss = (y_pred - y)^2，所以梯度是2 * (y_pred - y)，这是在对y_pred求导
    # 2. 接下来再把y_pred的形式带入，对a, b, c, d分别求导，就得到了下面的梯度表达式
    # 这里的梯度计算是手动推导的，后面换到PyTorch会自动计算梯度
    
    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 进行梯度下降，更新权重
    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
```

看看人家，numpy也能写神经网络

### 2. torch: tensor example

numpy没法用GPU加速，也没法自动求导，但torch可以！

**新概念：张量（tensor）**

（翻译）PyTorch 张量在概念上与 numpy 数组完全一致：张量是 n 维数组，PyTorch 为这些张量提供了丰富的运算函数。在后台，张量能够追踪计算图和梯度，同时也是科学计算的通用工具。

TLDR：张量就是多维数组，类似numpy的ndarray，但可以用GPU加速计算，并且可以自动求导。

> 注意一定要**明确指定**计算设备device，是CPU还是GPU，默认是CPU

[code](./2.py)

```python
# -*- coding: utf-8 -*-

import torch
import math


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```