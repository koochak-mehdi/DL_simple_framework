from Layers import Base

import numpy as np

class CrossEntropyLoss(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.eps = np.finfo(float).eps
        return

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(-np.log(input_tensor[label_tensor == 1] + self.eps))

    def backward(self, label_tensor):
        return -np.divide(label_tensor, self.input_tensor)