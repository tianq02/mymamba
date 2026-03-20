from fontTools.merge.util import current_time

# pytorch speedrun

1. Install PyTorch

    听我的，用docker环境预装的，别没事找事

2. 很抱歉我现在屁都不会，接下来我们逐个示例程序看

   参考：https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html

## hints:

1. 有关进度条：

我个人推荐用tqdm，它在终端永远不会出问题，在notebook中只要改成tqdm.notebook就行

在notebook中，如果训练较快，疯狂写loss信息到进度条可能会爆炸，需要手动限制刷新频率，例如：

```python
import time
from tqdm import tqdm

epoch = 10000
update_interval = 0.5  # 秒

bar = tqdm(total=epoch)
update_next = time.time()

for t in range(1,epoch+1):
   time.sleep(0.001)
   bar.update(1)
   
   current_time = time.time()
   if current_time > update_next:
      update_next = current_time + update_interval
      bar.set_postfix(loss="current loss here")
```

见[bar](./bar.ipynb)

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

# 我们的主机上是一块RTX 5070ti laptop, 它作为消费级GPU，浮点性能被砍得很厉害
# 所以这里的演示中未必能体现出GPU加速的优势
# 但推理上，将模型量化成8位整数后，GPU的推理性能会大幅提升
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# 随机输入输出数据，和numpy例子中一样
# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 随机初始化三次多项式的系数（权重）
# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

# pytorch可以自动化求导，梯度下降，但这里我们还是手动实现一遍

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

### 3. torch: autograd example

torch的自动求导功能，可以帮我们省去手动计算梯度的麻烦

> PyTorch: Tensors and autograd
> In the above examples, we had to manually implement both the forward and backward passes of our neural network. Manually implementing the backward pass is not a big deal for a small two-layer network, but can quickly get very hairy for large complex networks.
>
> Thankfully, we can use automatic differentiation to automate the computation of backward passes in neural networks. The autograd package in PyTorch provides exactly this functionality. When using autograd, the forward pass of your network will define a computational graph; nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors. Backpropagating through this graph then allows you to easily compute gradients.
>
> This sounds complicated, it’s pretty simple to use in practice. Each Tensor represents a node in a computational graph. If x is a Tensor that has x.requires_grad=True then x.grad is another Tensor holding the gradient of x with respect to some scalar value.
>
> Here we use PyTorch Tensors and autograd to implement our fitting sine wave with third order polynomial example; now we no longer need to manually implement the backward pass through the network:


TLDR:

1. 使用autograd，前向传播时自动生成计算图，后续调用backward()即可自动计算梯度
2. 需要给参数张量设置requires_grad=True，表示需要计算梯度
3. 计算出的梯度会存储在参数张量的.grad属性中

这里的代码和前面有不少区别，思想有点类似“懒惰求值”，先把前向传播的计算图记录下来，等需要梯度时再统一计算
你会发现：我们只定义了y_pred和loss的计算方式，梯度的计算完全交给了autograd

这真的太爽了，试想下，训练一个新模型，只要换下y_pred的定义就行了，根本不用管梯度怎么算。甚至说不定连loss都不用管，直接用pytorch内置的loss函数就行了


[code](./3.py)

```python
import torch
import math

# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

dtype = torch.float
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)

# 理解：这里我们在定义计算图而不是直接计算数值
# 下面的xyabcd都是张量（tensor），或者说计算图上的节点
# 梯度下降计算中不需要对x和y求导，所以它们的requires_grad=False，默认即可
# 而a,b,c,d是我们要优化的参数，需要求导，所以它们的requires_grad=True
# **不要忘了梯度下降是在对loss函数求导，而不是对y_pred求导**

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-1, 1, 2000, dtype=dtype)
y = torch.exp(x) # A Taylor expansion would be 1 + x + (1/2) x**2 + (1/3!) x**3 + ...

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

initial_loss = 1.
learning_rate = 1e-5
for t in range(5000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()

    # 记录初始loss
    # 不是必须的，只是说我们想展示每次迭代loss相比上一次的变化
    # Calculate initial loss, so we can report loss relative to it
    if t==0:
        initial_loss=loss.item()

    if t % 100 == 99:
        print(f'Iteration t = {t:4d}  loss(t)/loss(0) = {round(loss.item()/initial_loss, 6):10.6f}  a = {a.item():10.6f}  b = {b.item():10.6f}  c = {c.item():10.6f}  d = {d.item():10.6f}')

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 必须清零梯度，否则下次backward时梯度会累加
        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

### 4. torch: custom autograd

对于大型模型，直接处理梯度还是太底层了。

构建神经网络时，我们常将计算划分为多个层级，其中部分层包含可学习参数，这些参数将在训练过程中进行优化。

nn包定义了一组模块（Module），大致相当于神经网络层。模块接收输入张量并计算输出张量，同时可能保存内部状态（如包含可学习参数的张量）。该包还定义了训练神经网络时常用的实用损失函数。

TLDR：现在我们连模型也不自己定义了，而是搭积木一样

[code](./4.py)

[notebook](./4x.ipynb) 还有神秘自动次数功能

```python
# -*- coding: utf-8 -*-
import torch
import math


# 生成训练数据
# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 把y看作x幂次的线性组合，变成一个线性层神经网络
# unsqueeze是在指定位置插入一个维度,然后pow是逐元素幂运算

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flattens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1), # 线性组合x的各次幂，即前面的xx各列
    torch.nn.Flatten(0, 1) # 展平输出，这里单个输出相当于求和
)

# MSE就是均方误差（Mean Squared Error）
# 它的效果和前面我们手动实现的平方误差是一样的，但更不容易出现loss爆炸
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

# 对于这里仅求和的MSE，更大的数据量会带来数值上更大的loss
# 很有可能会导致nan，届时什么梯度都求不出来
# 解决方法：降低学习率
# 如果发现降低学习率之后，训练后半又下降特别慢，去看下自适应学习率的工作
learning_rate = 1e-6
for t in range(2000):

    # 前向传播：将xx（提前计算的x各次幂）传入模型，计算预测的y
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(xx)

    # 计算损失，这里的loss_fn是nn.MSELoss实例，是前面我们定义的均方误差函数
    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 梯度清零，否则梯度会累加
    # **重要** 之前手搓的程序不收敛就是这里漏了
    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # 反向传播：计算损失相对于模型中所有可学习参数的梯度
    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # 梯度下降，更新权重
    # 这里我们直接遍历模型的所有参数，然后根据它们的梯度更新，不再需要手动一个个更新
    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]

# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

### 5. optim example

想法很简单：

- 前面每次更新权重的时候都要手动地暂停梯度计算然后逐个更新权重，这要写三行
- 而且，注意到：随着训练的进行，似乎需要更高学习率来让 loss 下降更快，现有的固定学习率梯度下降效果不是很好

解决：

- 将优化过程包装成 optim 下面的模块，每次不再需要写三行，只要一句`optimizer.step()`就可以了

[code](./5x.ipynb) 还有船新误差绘图展示

注：这里的代码是我自己乱写的，可能有问题

```python
import time

import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(dtype)
torch.set_default_device(device)

x = torch.linspace(-2,2,10000)
y = x.sigmoid()

plt.plot(x.detach().cpu().numpy(),y.detach().cpu().numpy())
plt.show()

epoch = 100000
learning_rate = 2e-3
max_pow = 6
msg_precision = 3
update_interval = 0.5

p = torch.tensor(range(1, max_pow + 1))
xx = x.unsqueeze(-1).pow(p)

loss_fn = torch.nn.MSELoss()

model = torch.nn.Sequential(
   torch.nn.Linear(max_pow, 1),
   torch.nn.Flatten(0, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

with tqdm(total=epoch) as bar:
   update_next = time.time()

   for i in range(1, epoch + 1):
      y_pred = model(xx)
      loss = loss_fn(y_pred, y)

      bar.update(1)

      if time.time() > update_next or i == epoch:
         update_next += update_interval
         bar.set_postfix(loss=loss.item())

      # 注意这里的zero_grad在optimizer上，理论上直接在model上zero_grad也行，不更新优化器内部的状态可能造成问题
      optimizer.zero_grad()

      loss.backward()

      
      # 原先手动的一大堆更新，现在只要简单地一个step
      # with torch.no_grad():
      #     for param in model.parameters():
      #         param -= learning_rate * param.grad
      optimizer.step()

msg = f"y = {model[0].bias.item():.{msg_precision}f}"
for p, w in enumerate(model[0].weight.tolist()[0], start=1):
   if abs(w) < pow(0.1, msg_precision):
      continue
   msg += " - " if w < 0 else " + "
   msg += f"{abs(w):.{msg_precision}f}"
   msg += f" x^{p}" if p > 1 else " x"
print(msg)
```

### 6. custom nn modules:

自定义 nn 模块：

有时，要构建的模型不只是简单地序列组合现有模块，这时可以通过编写`nn.Module`的子类来自定义模块，仅需重写其中的`forward`方法即可

> 在mamba-minimal中大量看到了这种写法，必须掌握

[sample](6.ipynb)

[code](6x.ipynb) 含有手搓的终极版格式设置，以及用sympy的latex渲染输出

```python
# -*- coding: utf-8 -*-
import torch
import math


class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```