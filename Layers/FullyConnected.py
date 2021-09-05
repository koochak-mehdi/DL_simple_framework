import numpy as np

from . import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size+1, output_size))
        #self.weights = np.random.sample((output_size, input_size+1))
        self._optimizer = None
        self.gradient_weights = None
    
    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor,           # add bias to input_tensor
                                            np.ones((input_tensor.shape[0], 1))), 
                                            axis=1)
        #print('input_size -- ', input_tensor.shape)
        #print('input -- ', self.input_size)
        #print('w -- ', self.weights.shape)
        #print('output -- ', self.output_size)
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self.error_tensor =error_tensor
        self.gradient_weights = self.get_gradient_weights()

        if self._optimizer is not None:
            self.weights = self.get_optimizer().calculate_update(self.weights,
                                                                self.gradient_weights)

        return np.dot(error_tensor, self.weights.T)[:,:-1]    # remove bias
    

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), 
                                                    self.input_size, 
                                                    self.output_size)
        bias = bias_initializer.initialize((1, self.output_size),
                                            self.input_size,
                                            self.output_size)
        #self.weights = np.concatenate((weights, bias))
        self.weights = np.vstack((weights, bias))
        pass

    def get_gradient_weights(self):
        return np.dot(self.input_tensor.T, self.error_tensor)
    

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        return
    
    def get_optimizer(self):
        return self._optimizer
    
    optimizer = property(get_optimizer, set_optimizer)
