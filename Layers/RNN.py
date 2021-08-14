import numpy as np
import copy
from . import Base
from . import FullyConnected, TanH, Sigmoid


class RNN(Base.BaseLayer):
    def __init__(self , input_size, hidden_size, output_size):
        super().__init__()
        self.trainable= True
        self.input_size= input_size
        self.hidden_size= hidden_size
        self.output_size= output_size
        self.hidden_state= np.zeros((1,hidden_size))
        self._memorize= False
        self.h_t = None
        self.input_tensor_memory_hidden= list()
        self.input_tensor_memory_output= list()
        self.activation_tanh= list()
        self.activation_sigmoid= list()
        self._gradient_w = None
        self._gradient_wy = None
        self._optimizer_output= None
        self._optimizer_hidden= None

        self.full_conn_hidden= FullyConnected.FullyConnected(input_size= input_size+hidden_size,
                                                                output_size= hidden_size)
        self.full_conn_output= FullyConnected.FullyConnected(input_size= hidden_size, 
                                                                output_size= output_size)
        self.sigmoid= Sigmoid.Sigmoid()
        self.tanh= TanH.TanH()

    def get_memorize(self):
        return self._memorize

    def set_memorize(self,memorize):
        self._memorize= memorize

    memorize= property(get_memorize, set_memorize)
 
    def forward(self, input_tensor):
        self.input_tensor= input_tensor
        self.T= input_tensor.shape[0]
        output= np.zeros((self.T, self.output_size))

    
        for t in range(self.T):
            #print("input________shape", input_tensor[t].shape)
            #print("hidden__________shape", self.hidden_state.shape)
            self.x_tild= np.hstack(( self.hidden_state, 
                                    self.input_tensor[t].reshape(1, input_tensor[t].shape[0])))

            self.input_tensor_memory_hidden.append(self.x_tild)

            u_t = self.full_conn_hidden.forward(self.x_tild)
            self.hidden_state= self.tanh.forward(u_t)
            self.activation_tanh.append(self.tanh.activition)

            self.input_tensor_memory_output.append(self.hidden_state)
            o_t= self.full_conn_output.forward(self.hidden_state)
            output[t]= self.sigmoid.forward(o_t)
            self.activation_sigmoid.append(self.sigmoid.activition)
        self.h_t= self.hidden_state

        if self._memorize:
            self.hidden_state = self.h_t
        else:
            self.hidden_state = np.zeros((1,self.hidden_size) )

        return output

    def backward(self, error_tensor):
        gradient_hidden_2= np.zeros_like(self.hidden_state)
        output_plane = np.zeros((self.T, self.input_size))
        self._gradient_w = np.zeros_like(self.full_conn_hidden.weights)
        self._gradient_wy = np.zeros_like(self.full_conn_output.weights)

        for t in reversed(range(self.T)):
            gradient_hidden_1= self.full_conn_output.backward(self.sigmoid.backward(error_tensor[t]))
            self.full_conn_output.input_tensor = np.hstack((self.input_tensor_memory_output[t-1],
                                                            np.ones((1,1))))
            self.sigmoid.activition= self.activation_sigmoid[t-1]

            gradient_hidden_state= gradient_hidden_2 + gradient_hidden_1

            gradient_x_tild= self.full_conn_hidden.backward( self.tanh.backward(gradient_hidden_state))
            self.full_conn_hidden.input_tensor = np.hstack((self.input_tensor_memory_hidden[t-1],
                                                            np.ones((1,1))))
            self.tanh.activition = self.activation_tanh[t-1]
            gradient_hidden_2, gradent_x = np.split(gradient_x_tild, [self.hidden_size], axis=1)
            output_plane[t] = gradent_x
        
            self._gradient_w += self.full_conn_hidden.gradient_weights

            self._gradient_wy += self.full_conn_output.gradient_weights

        if self._optimizer_output is not None:
            self.full_conn_output.weights= self._optimizer_output.calculate_update(self.full_conn_output.weights,
                                                                                    self._gradient_wy)

        if self._optimizer_hidden is not None:
            self.full_conn_hidden.weights = self._optimizer_hidden.calculate_update(self.full_conn_hidden.weights,
                                                                                    self._gradient_w)

        return output_plane


    def get_optimizer(self):
        return self._optimizer_output, self._optimizer_hidden

    def set_optimizer(self,  opt):
        self._optimizer_output = opt
        self._optimizer_hidden = copy.deepcopy(opt)

    optimizer = property(get_optimizer,set_optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.full_conn_hidden.initialize(weights_initializer, bias_initializer)
        self.full_conn_output.initialize(weights_initializer,bias_initializer)

    def get_weights(self):
        return self.full_conn_hidden.weights
    def set_weights(self, weights):
        self.full_conn_hidden.weights = weights

    weights = property(get_weights, set_weights)

    def get_gradient_weights(self):
        return self._gradient_w
    
    gradient_weights = property(get_gradient_weights)