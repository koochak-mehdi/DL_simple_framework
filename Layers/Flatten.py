from . import Base
import numpy as np 

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.shape=input_tensor.shape
        return np.reshape(input_tensor, (self.shape[0], np.prod(self.shape[1:])))
         
    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.shape)