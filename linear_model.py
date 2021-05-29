import numpy as np
from nadl.tensor import Tensor


x = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype=np.float))
y = x @ coef + 5

w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)
learning_rate = 0.001
batch_size = 32


for epoch in range(100):
    epoch_loss = 0.0

    w.zero_grad()
    b.zero_grad()
    predicted = x @ w + b
    errors = predicted - y
    loss = (errors * errors).sum()
    loss.backward()
    epoch_loss += loss.data

    w -= w.grad * learning_rate
    b -= b.grad * learning_rate
    print(epoch, epoch_loss)
