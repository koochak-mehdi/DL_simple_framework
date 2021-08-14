from Optimization import Constraints
import numpy as np

class Optimizer:
    def __init__(self) -> None:
        self.regularizer = None
        pass
    def add_regularizer(self,regularizer):
        self.regularizer= regularizer

class Sgd(Optimizer): 
    def __init__(self, learning_rate):
        self.learning_rate= learning_rate
        super().__init__()

    def add_regularizer(self, regularizer):
        return super().add_regularizer(regularizer)

    def calculate_update (self, weight_tensor, gradient_tensor):
        constraint = 0
        if self.regularizer is not None : 
            constraint = self.regularizer.calculate_gradient(weight_tensor)

        uploeaded_weight= (weight_tensor-constraint* self.learning_rate) - np.dot(self.learning_rate, gradient_tensor)


        return uploeaded_weight



class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate , momentum_rate):
        self.learning_rate=learning_rate
        self.momentum_rate= momentum_rate
        self.v=None
        super().__init__()
        pass

    def add_regularizer(self, regularizer):
        return super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        constraint = 0
        if self.regularizer is not None:
            constraint = self.regularizer.calculate_gradient(weight_tensor)

        if self.v is None:
        # self.v = np.zeros(weight_tensor.shape)
            self.v = np.zeros_like(weight_tensor)
        self.v= np.subtract(np.dot(self.momentum_rate , self.v), 
                        np.dot(self.learning_rate, gradient_tensor))

        return (weight_tensor - constraint* self.learning_rate) + self.v


class Adam(Optimizer):
    def __init__(self, learning_rate, mu , rho):
        self.learning_rate=learning_rate
        self.mu=mu
        self.rho=rho
        self.v=None
        self.r=None
        self.k=1
        self.epsilon= np.finfo(float).eps
        super().__init__()
    
    def add_regularizer(self, regularizer):
        return super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        constraint = 0
        if self.regularizer is not None:
            constraint = self.regularizer.calculate_gradient(weight_tensor)
        if self.v is None :
            self.v= np.dot((1-self.mu), gradient_tensor)
        else:
            self.v= np.dot(self.mu,self.v) + np.dot((1-self.mu), gradient_tensor)
        
        if self.r is None:
            self.r= np.dot((1-self.rho), np.square(gradient_tensor))
        else:
            self.r= np.dot(self.rho, self.r) + np.dot((1-self.rho), np.square(gradient_tensor))

        v_hat= np.divide(self.v, (1-np.power(self.mu, self.k)))
        r_hat= np.divide(self.r, (1-np.power(self.rho, self.k)))

        self.k+=1

        return (weight_tensor - self.learning_rate * constraint) - np.dot(self.learning_rate, 
                                np.divide(v_hat,np.sqrt(r_hat)+ self.epsilon))
    #weight = optimizer.calculate_update(weight, gradW)
    