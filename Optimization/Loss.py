
import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.epsilon= np.finfo(float).eps

    def forward(self, input_tensor, label_tensor):
        self.input_tensor=input_tensor
        loss= np.sum(-np.log(input_tensor[label_tensor==1] + self.epsilon))
        return loss


    def backward(self, label_tensor):

        return -label_tensor/self.input_tensor
