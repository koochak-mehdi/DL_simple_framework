from . import Base
import numpy as np
from . import Helpers
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels

        self.weights = np.ones((channels, 1))
        self.bias = np.zeros((channels, 1))

        self._gradient_weights = None
        self._gradient_bias = None

        self.weight_optimizer = None
        self.bias_optimizer = None
        
        self.k = 0
        self.eps = np.finfo(float).eps
        self.alpha = .8
    
    def initialize(self, weights_initializer, bias_initializer):
        """self.weights = weights_initializer.initialize(self.weights.shape,
                                                        self.eps,
                                                        self.eps)
        self.bias = bias_initializer.initialize(self.bias.shape,
                                                    self.eps,
                                                    self.eps)"""
        self.weights = np.ones((self.channels, 1))
        self.bias = np.zeros((self.channels, 1))                                                    
        pass


    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 2:    #FC
            output_tensor = np.zeros_like(input_tensor)
            if self.testing_phase is False: # training
                self.mu_b = np.mean(input_tensor, axis=0).reshape((input_tensor.shape[1], 1))
                self.sigma_b = np.var(input_tensor, axis=0).reshape((input_tensor.shape[1], 1))
                self.x_tilde = (input_tensor - self.mu_b.T) / np.sqrt(self.sigma_b.T + self.eps)
                output_tensor = self.weights.T * self.x_tilde + self.bias.T
            else:                           # testing
                if self.k == 0:
                    self.mu = (1-self.alpha)*self.mu_b
                    self.sigma = (1-self.alpha)*self.sigma_b
                else:
                    self.mu = self.alpha*self.mu + (1-self.alpha)*self.mu_b
                    self.sigma = self.alpha*self.sigma + (1-self.alpha)*self.sigma_b
                self.x_tilde = (input_tensor - self.mu_b.T) / np.sqrt(self.sigma_b.T + self.eps)
                output_tensor = self.weights.T * self.x_tilde + self.bias.T
        else:                               #CNN
            self.B, self.H, self.M, self.N = input_tensor.shape
            modified_input_tensor = self.reformat(input_tensor)
            output_tensor = np.zeros_like(modified_input_tensor)
            if self.testing_phase is False: #training
                self.mu_b = np.mean(modified_input_tensor, axis=0).reshape(modified_input_tensor.shape[1],1)
                self.sigma_b = np.var(modified_input_tensor, axis=0).reshape(modified_input_tensor.shape[1],1)
                self.x_tilde = (modified_input_tensor - self.mu_b.T) / np.sqrt(self.sigma_b.T + self.eps)
                output_tensor = self.weights.T * self.x_tilde + self.bias.T
                output_tensor = self.reformat(output_tensor)
            else:                           #testing
                if self.k == 0:
                    self.mu = (1-self.alpha)*self.mu_b
                    self.sigma = (1-self.alpha)*self.sigma_b
                else:
                    self.mu = self.alpha*self.mu + (1-self.alpha)*self.mu_b
                    self.sigma = self.alpha*self.sigma + (1-self.alpha)*self.sigma_b
                self.x_tilde = (modified_input_tensor - self.mu_b.T) / np.sqrt(self.sigma_b.T + self.eps)
                output_tensor = (self.weights.T* self.x_tilde) + self.bias.T
                output_tensor = self.reformat(output_tensor)

        return output_tensor
        

    def backward(self, error_tensor):
        flag = False
        input_tensor = self.input_tensor
        if len(error_tensor.shape) != 2:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
            flag = True
        
        self._gradient_bias = np.sum(error_tensor, axis=0).reshape(self.bias.shape)
        self._gradient_weights = np.sum(error_tensor * self.x_tilde, axis=0).reshape(self.weights.shape)
        self.gradient_input = Helpers.compute_bn_gradients(error_tensor,
                                                                input_tensor,
                                                                self.weights.T,
                                                                self.mu_b.T,
                                                                self.sigma_b.T,
                                                                self.eps)
        
        if self.weight_optimizer is not None:
            self.weights = self.weight_optimizer.calculate_update(self.weights, self._gradient_weights)

        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        
        if flag:
            self.gradient_input = self.reformat(self.gradient_input)

        return self.gradient_input

    def reformat(self, tensor):
        if len(tensor.shape) == 2:  # flatten to CNN
            reformat_tensor = tensor.reshape(self.B, self.M*self.N, self.H)
            reformat_tensor = np.swapaxes(reformat_tensor, 1,2)
            reformat_tensor = reformat_tensor.reshape(self.B, self.H, self.M, self.N)
        else:                       # CNN to flatten
            self.B,self.H,self.M,self.N = tensor.shape
            reformat_tensor = tensor.reshape(self.B,self.H,self.M*self.N)
            reformat_tensor = np.swapaxes(reformat_tensor, 1,2)
            reformat_tensor = reformat_tensor.reshape(self.B*self.M*self.N,self.H)

        return reformat_tensor
    
    def get_optimizer(self):
        return self.weight_optimizer, self.bias_optimizer
    
    def set_optimizer(self, optmizer):
        self.weight_optimizer = optmizer
        self.bias_optimizer = copy.deepcopy(optmizer)

    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias

    optimizer = property(get_optimizer, set_optimizer)
    gradient_weights = property(get_gradient_weights)
    gradient_bias = property(get_gradient_bias)