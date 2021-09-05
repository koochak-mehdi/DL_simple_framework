from . import Base
import numpy as np

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1-self.activation**2)
