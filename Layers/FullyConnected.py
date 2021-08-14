from . import Base 
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size= input_size
        self.output_size= output_size
        
        self.trainable= True
        self.weights = np.random.uniform(0,1, (input_size+1,output_size))
        self._optimizer = None
        self.gradient_weights = None
        return
        
    def forward(self, input_tensor):
        to_append = np.ones((input_tensor.shape[0],1))
        self.input_tensor = np.append(input_tensor, to_append, axis=1)
         
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        
        if self._optimizer is not None:
            opt = self.get_optimizer()
            self.weights = opt.calculate_update(self.weights, self.gradient_weights)

        err_back = np.dot(error_tensor,self.weights.T)
        return err_back[:, :-1]


    def initialize(self, weights_initializer, bias_initializer):
        weight= weights_initializer.initialize((self.input_size,self.output_size),
                                                self.input_size,
                                                self.output_size)
        bias= bias_initializer.initialize((1,self.output_size),
                                            self.input_size,
                                            self.output_size)
        self.weights= np.vstack((weight,bias))
        pass


    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)


  