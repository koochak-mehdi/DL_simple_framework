from . import Base 
import numpy as np

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor - np.max(input_tensor)
        self.y_hat= np.exp(self.input_tensor) / np.sum(np.exp(self.input_tensor), axis=1)[:,np.newaxis]
        return self.y_hat

    def backward(self, error_tensor):
        return self.y_hat*(error_tensor- np.sum(np.multiply(error_tensor, self.y_hat), axis=1)[:,np.newaxis] )
        