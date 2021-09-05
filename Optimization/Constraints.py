import numpy as np

class L2_Regularizer:
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha*weights
    
    def norm(self, weights):
        return self.alpha*np.sum(weights**2)

class L1_Regularizer:
    def __init__(self, alpha) -> None:
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        return self.alpha*np.sign(weights)

    def norm(self, weights):
        return self.alpha*np.sum(np.absolute(weights))