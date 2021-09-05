from . import Base

import numpy as np

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.error_tensor = None
        return

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(input_tensor, 0)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor * (self.input_tensor > 0)
        return self.error_tensor 