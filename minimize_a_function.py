from nadl.tensor import Tensor

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

# we want to minimize the sum of squares
for i in range(100):
    x.zero_grad()
    sum_of_squares = (x * x).sum()
    sum_of_squares.backward()
    delta_x = 0.1 * x.grad
    x -= delta_x
    print(i, sum_of_squares)
# print(x.zero_grad())
