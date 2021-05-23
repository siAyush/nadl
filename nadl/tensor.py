from typing import List
import numpy as np


class Tensor():
    def __init__(self,
                 data: np.ndarray,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.depends_on = depends_on
        self.shape = data.shape
