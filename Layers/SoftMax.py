from . import Base

import numpy as np

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        return

    def forward(self, input_tensor):
        self.input_tensor = input_tensor - np.max(input_tensor)


        # have a question here!!
        # why it works?! 
        # which dimension is batch
        self.y_hat = np.divide(np.exp(self.input_tensor),
                        np.sum(np.exp(self.input_tensor), axis=1)[:, np.newaxis])

        return self.y_hat

    def backward(self, error_tensor):
        sec_term = np.subtract(error_tensor,
                                np.sum(np.multiply(error_tensor, self.y_hat), axis=1)[:, np.newaxis])
        return self.y_hat * sec_term