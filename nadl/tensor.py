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
        self.data = data
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = data.shape
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
