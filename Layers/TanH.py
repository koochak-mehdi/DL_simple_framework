from . import Base
import numpy as np 

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.activition= np.tanh(input_tensor)
        return self.activition
        

    def backward(self , error_tensor):
        return error_tensor * (1- np.square(self.activition))
        

