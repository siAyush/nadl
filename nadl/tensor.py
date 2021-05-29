from typing import Callable, List
import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fun: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union["Tensor", float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> "Tensor":
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor():
    def __init__(self,
                 data: np.ndarray,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional["Tensor"] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def sum(self) -> "Tensor":
        return tensor_sum(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> "Tensor":
        "t + other"
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> "Tensor":
        "other + t"
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> "Tensor":
        pass

    def __mul__(self, other) -> "Tensor":
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        return _mul(ensure_tensor(other), self)

    def __imul__(self, other) -> "Tensor":
        pass

    def __neg__(self) -> "Tensor":
        return _negative(self)

    def __sub__(self, other) -> "Tensor":
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other) -> "Tensor":
        pass

    def backward(self, grad: "Tensor" = None) -> None:
        assert self.requires_grad, "called backward on non requires grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fun(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0 dimensional tensor
    tensor that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fun(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)
        depends_on = [Dependency(t, grad_fun)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fun1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t1, grad_fun1))

    if t2.requires_grad:
        def grad_fun2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependency(t2, grad_fun2))

    return Tensor(data, requires_grad, depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def _negative(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + (-t2)
