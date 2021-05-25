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
        self.grad = Tensor(np.zeros_like(self.data))

    def sum(self) -> "Tensor":
        return tensor_sum(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

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


def tensor_sum(tensor: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0 dimensional tensor
    tensor that's the sum of all its elements.
    """
    data = tensor.data.sum()
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fun(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(tensor.data)
        depends_on = [Dependency(tensor, grad_fun)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
