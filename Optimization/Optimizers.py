from typing import ContextManager
import numpy as np

class Optimizer:
    def __init__(self) -> None:
        self.regularizer = None
        pass
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        super().__init__()

    def add_regularizer(self, regularizer):
        return super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        cons_term = 0
        if self.regularizer is not None:
            cons_term = self.regularizer.calculate_gradient(weight_tensor)
            
        return weight_tensor - cons_term* self.learning_rate - np.dot(self.learning_rate, gradient_tensor)


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.cache = None
        super().__init__()

    def add_regularizer(self, regularizer):
        return super().add_regularizer(regularizer)        

    def calculate_update(self, weight_tensor, gradient_tensor):
        cons_term = 0
        if self.regularizer is not None:
            cons_term = self.regularizer.calculate_gradient(weight_tensor)

        if self.cache is None:
            self.cache = np.zeros_like(weight_tensor)
        
        v = np.subtract(np.dot(self.momentum_rate, self.cache),
                        np.dot(self.learning_rate, gradient_tensor))

        self.cache = v
        return weight_tensor - self.learning_rate*cons_term + v


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.cache = dict()
        self.eps = np.finfo(float).eps
        super().__init__()
        pass

    def add_regularizer(self, regularizer):
        return super().add_regularizer(regularizer)

    def calculate_update(self, weight_tensor, gradient_tensor):
        cons_term = 0
        if self.regularizer is not None:
            cons_term = self.regularizer.calculate_gradient(weight_tensor)

        if len(self.cache) == 0:
            self.cache['v'] = np.zeros_like(weight_tensor)
            self.cache['r'] = np.zeros_like(weight_tensor)
            self.cache['k'] = 1

        v = np.dot(self.mu, self.cache['v']) + \
                np.dot((1 - self.mu), gradient_tensor)
        r = np.dot(self.rho, self.cache['r']) + \
                np.dot((1 - self.rho), np.square(gradient_tensor))

        v_hat = np.divide(v, (1 - np.power(self.mu, self.cache['k'])))
        r_hat = np.divide(r, (1 - np.power(self.rho, self.cache['k'])))

        self.cache['v'] = v
        self.cache['r'] = r
        self.cache['k'] += 1

        return np.subtract(weight_tensor - cons_term*self.learning_rate, 
                        np.dot(self.learning_rate, (np.divide(v_hat, 
                                                        np.sqrt(r_hat) + self.eps))))
