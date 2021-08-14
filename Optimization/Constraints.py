import numpy as np

class L1_Regularizer:
    def __init__(self,alpha) -> None:
        self.alpha=alpha
        pass
    
    def norm(self,weights):
        return  np.sum(np.absolute(weights))* self.alpha

    def calculate_gradient(self, weights):
        return np.sign(weights) *self.alpha
    

class L2_Regularizer:
    def __init__(self, alpha) -> None:
        self.alpha=alpha
        pass
    def calculate_gradient(self, weights):
        return weights*self.alpha
        

    def norm(self,weights):
        return self.alpha * np.sum(weights**2)
