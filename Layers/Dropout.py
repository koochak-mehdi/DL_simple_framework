from . import Base
import numpy as np

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase is False:
            # row, col = input_tensor.shape
            self.dropout_mask = np.random.rand(*input_tensor.shape) < self.probability
            return input_tensor * self.dropout_mask / self.probability
        else:
            return input_tensor
            
    def backward(self, error_tensor):
        return error_tensor * self.dropout_mask / self.probability