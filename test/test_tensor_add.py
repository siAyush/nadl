import unittest
from nadl.tensor import Tensor


class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)
        t3 = t1 + t2
        assert t3.data.tolist() == [5, 7, 9]
        t3.backward(Tensor([-1., -2., -3.]))

        assert t1.grad.data.tolist() == [-1, -2, -3]
        assert t2.grad.data.tolist() == [-1, -2, -3]

    def test_broadcast_add(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([7, 8, 9], requires_grad=True)
        t3 = t1 + t2
        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)
        t3 = t1 + t2
        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]
