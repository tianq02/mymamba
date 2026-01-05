import torch

from tqdm import tqdm

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(dtype)
torch.set_default_device(device)

x = torch.linspace(-1,1, 5000)
y = torch.sin(x)

a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)
e = torch.randn((), requires_grad=True)

learning_rate = 1e-5
epoch = 10000

with tqdm(total = epoch) as pbar:
    for t in range(epoch):
        y_pred = a + b * x + c * x**2 + d * x**3 + e * x**4
        loss = (y_pred - y).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad
            e -= learning_rate * e.grad

            a.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()
            d.grad.zero_()
            e.grad.zero_()

        pbar.update(1)
        pbar.set_postfix({'loss': loss.item()})

print(f"y = {a} + {b} x+ {c} x^2 + {d} x^3 + {e} x^4")

y_pred = a + b * x + c * x**2 + d * x**3 + e * x**4