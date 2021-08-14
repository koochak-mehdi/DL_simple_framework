import numpy as np 
from. import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward( self, input_tensor):
        self.activition = 1/(1+ np.exp(-input_tensor))
        return self.activition

    def backward(self, error_tensor):
        return error_tensor * (1- self.activition)*self.activition