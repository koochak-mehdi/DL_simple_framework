import numpy as np
from . import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability) -> None:
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase is False:
            self.mask= np.random.rand(*input_tensor.shape) < self.probability
            return input_tensor * self.mask / self.probability 
        else:
            return input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.mask / self.probability