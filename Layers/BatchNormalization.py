from unittest.case import TestCase
import numpy as np
from numpy.core.fromnumeric import mean, reshape
from numpy.core.numeric import outer
from numpy.lib.function_base import select
from . import Base
from . import Helpers
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights = np.ones((channels, 1)) #gamma
        self.bias = np.zeros((channels, 1)) #beta
        self.trainable = True
        self.epsilon = np.finfo(float).eps
        self.k = 0
        self.alpha = .8

        self._gradien_weight = None
        self._gradient_bias = None
        self.weight_optimizer = None
        self.bias_optimizer = None


    def forward(self, input_tensor):
        
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 2:   #fullyCOn
            output_tensor = np.zeros_like(input_tensor)

            if self.testing_phase is False:  #traningPhase
                self.mean_b = np.mean(input_tensor, axis=0).reshape((input_tensor.shape[1], 1))
                self.sigma_b= np.var(input_tensor,axis=0).reshape((input_tensor.shape[1], 1))
                self.x_tild = ( input_tensor - self.mean_b.T) / np.sqrt( self.sigma_b.T+ self.epsilon)
                output_tensor = self.weights.T * self.x_tild + self.bias.T
            else: #testPhase
                if self.k == 0:
                    self.mean = (1 - self.alpha) * self.mean_b
                    self.sigma = (1- self.alpha)* self.sigma_b
                else:
                    self.mean= self.alpha * self.mean + (1 - self.alpha) * self.mean_b
                    self.sigma= self.alpha * self.sigma + (1- self.alpha)* self.sigma_b
                self.x_tild=  (input_tensor - self.mean_b.T) / np.sqrt( self.sigma_b.T+ self.epsilon)
                output_tensor= self.weights.T * self.x_tild + self.bias.T
        else:
            self.b, self.h , self.m, self.n= input_tensor.shape
            modified_input_tensor= self.reformat(input_tensor)
            output_tensor= np.zeros_like(modified_input_tensor)
            if self.testing_phase is False: # training
                self.mean_b= np.mean(modified_input_tensor, axis=0).reshape(modified_input_tensor.shape[1], 1)
                self.sigma_b= np.var(modified_input_tensor, axis=0).reshape(modified_input_tensor.shape[1], 1)
                self.x_tild= (modified_input_tensor - self.mean_b.T) / np.sqrt(self.sigma_b.T + self.epsilon)
                output_tensor= self.weights.T * self.x_tild + self.bias.T
                output_tensor = self.reformat(output_tensor) 
            else: #testing 
                if self.k == 0:
                    self.mean = (1 - self.alpha) * self.mean_b
                    self.sigma = (1- self.alpha)* self.sigma_b
                else:
                    self.mean= self.alpha * self.mean + (1 - self.alpha) * self.mean_b
                    self.sigma= self.alpha * self.sigma + (1- self.alpha)* self.sigma_b
                self.x_tild= np.divide( modified_input_tensor - self.mean_b.T , np.sqrt( self.sigma_b.T+ self.epsilon))
                output_tensor= self.weights.T * self.x_tild + self.bias.T
                output_tensor= self.reformat(output_tensor)
                
        return output_tensor


    def backward(self, error_tensor):
        input_tensor= self.input_tensor
        flag= False
        if len(error_tensor.shape) != 2: 
            error_tensor= self.reformat(error_tensor)
            input_tensor= self.reformat(self.input_tensor)
            flag = True
        
        self._gradient_bias= np.sum( error_tensor, axis= 0).reshape(self.bias.shape)
        self._gradien_weight= np.sum( error_tensor * self.x_tild , axis= 0).reshape( self.weights.shape)
        self.gradient_input= Helpers.compute_bn_gradients(error_tensor, 
                                                            input_tensor, 
                                                            self.weights.T, 
                                                            self.mean_b.T, 
                                                            self.sigma_b.T, 
                                                            self.epsilon)

        if self.bias_optimizer is not None:
            self.bias= self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        
        if self.weight_optimizer is not None:
            self.weights = self.weight_optimizer.calculate_update(self.weights, self._gradien_weight)
        
        if flag:
            self.gradient_input= self.reformat(self.gradient_input)
        
        return self.gradient_input

    
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((self.channels, 1))
        self.bias = np.zeros((self.channels, 1))

    def reformat(self, tensor):
        if len( tensor.shape)== 2:  #flatten to cnn
            reformed_tensor = tensor.reshape(self.b, self.m * self.n, self.h)
            reformed_tensor= np.swapaxes(reformed_tensor, 1, 2)
            reformed_tensor= reformed_tensor.reshape(self.b, self.h, self.m, self.n)

        else: #cnn to flatten
            self.b, self.h, self.m, self.n = tensor.shape
            reformed_tensor= tensor.reshape(self.b, self.h, self.m * self.n)
            reformed_tensor= np.swapaxes(reformed_tensor, 1,2 )
            reformed_tensor= reformed_tensor.reshape( self.b * self.m * self.n , self.h)

        return reformed_tensor

    def get_optimizer(self):
        return self.weight_optimizer , self.bias_optimizer
        

    def set_optimizer(self, optimizer):
        self.weight_optimizer = optimizer
        self.bias_optimizer= copy.deepcopy(optimizer)
        return

    def get_gradient_weights(self):
        return self._gradien_weight
    
    def get_gradient_bias(self):
        return self._gradient_bias

    optimizer= property(get_optimizer, set_optimizer)
    gradient_weights = property(get_gradient_weights)
    gradient_bias = property(get_gradient_bias)