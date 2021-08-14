from . import Base 
import numpy as np

class ReLU(Base.BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(input_tensor, 0)
       
    
    def backward(self, error_tensor):
        mask = self.input_tensor > 0
        return np.multiply(error_tensor, mask)