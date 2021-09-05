from . import Base
from . import FullyConnected, TanH, Sigmoid

import numpy as np
import copy


class RNN(Base.BaseLayer):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()

        self.trainable = True
        
        self.input_size     = in_size
        self.hidden_size    = hid_size
        self.output_size    = out_size

        self.hidden_state = np.zeros((1, hid_size))

        self._memorize = False

        self.FC_hidden = FullyConnected.FullyConnected(in_size + hid_size,
                                                        hid_size)
        self.FC_output = FullyConnected.FullyConnected(hid_size,
                                                        out_size)
        
        self.sigmoid    = Sigmoid.Sigmoid()
        self.tanh       = TanH.TanH()

        self.initial_hidden = None

        self.input_tensor_fc_hidden_memory  = list()
        self.input_tensor_fc_output_memory  = list()
        self.activation_sigmoid_memory      = list()
        self.activation_tanh_memory         = list()

        self._gradient_W    = None
        self._gradient_Wy   = None

        self._optimizer_output = None
        self._optimizer_hidden = None
        
        
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.T = input_tensor.shape[0]
        output = np.zeros((self.T, self.output_size))

        if self._memorize:
            if self.initial_hidden is None:
                self.hidden_state = np.zeros((1, self.hidden_size))
            else:    
                self.hidden_state = self.initial_hidden
        else:
            self.hidden_state = np.zeros((1, self.hidden_size))

        for t in range(self.T):
            #print('### {} ###'.format(t))
            x_tilde = np.hstack((self.hidden_state, 
                                input_tensor[t].reshape(1, input_tensor[t].shape[0])))
            
            self.input_tensor_fc_hidden_memory.append(x_tilde)
            u_t = self.FC_hidden.forward(x_tilde)
            self.hidden_state = self.tanh.forward(u_t)
            self.activation_tanh_memory.append(self.tanh.activation)
            
            self.input_tensor_fc_output_memory.append(self.hidden_state)
            o_t = self.FC_output.forward(self.hidden_state)
            output[t] = self.sigmoid.forward(o_t)
            self.activation_sigmoid_memory.append(self.sigmoid.activation)

        self.initial_hidden = self.hidden_state
        return output

    def backward(self, error_tensor):
        #print('error_tensor -- ', error_tensor.shape)
        output = np.zeros((self.T, self.input_size))
        
        gr_hidden_2 = np.zeros_like(self.hidden_state)

        self._gradient_Wy = np.zeros_like(self.FC_output.weights)        
        self._gradient_W = np.zeros_like(self.FC_hidden.weights)
        

        for t in reversed(range(self.T)):
            #print('### {} ###'.format(t))
            gr_hidden_1 = self.FC_output.backward(self.sigmoid.backward(error_tensor[t]))
            self.FC_output.input_tensor = np.hstack((self.input_tensor_fc_output_memory[t-1],
                                                        np.ones((1,1))))
            self.sigmoid.activation = self.activation_sigmoid_memory[t-1]
            
            gr_hidden = gr_hidden_1 + gr_hidden_2
            gr_x_tilde = self.FC_hidden.backward(self.tanh.backward(gr_hidden))
            self.FC_hidden.input_tensor = np.hstack((self.input_tensor_fc_hidden_memory[t-1],
                                                        np.ones((1,1))))
            self.tanh.activation = self.activation_tanh_memory[t-1]                                                        
            
            gr_hidden_2, gr_input = np.split(gr_x_tilde, [self.hidden_size], axis=1)

            output[t] = gr_input

            self._gradient_Wy += self.FC_output.gradient_weights
            self._gradient_W += self.FC_hidden.gradient_weights
        
        if self._optimizer_output is not None:
            self.FC_output.weights = self._optimizer_output.calculate_update(self.FC_output.weights,
                                                                        self._gradient_Wy)

        if self._optimizer_hidden is not None:
            self.FC_hidden.weights = self._optimizer_hidden.calculate_update(self.FC_hidden.weights,
                                                                        self._gradient_W)
            

        return output
    
    def initialize(self, weights_initializer, bias_initializer):
        self.FC_output.initialize(weights_initializer, bias_initializer)
        self.FC_hidden.initialize(weights_initializer, bias_initializer)

    def set_memorize(self, mem):
        self._memorize = mem

    def get_memorize(self):
        return self._memorize
    
    def set_optimizer(self, opt):
        self._optimizer_output = opt
        self._optimizer_hidden = copy.deepcopy(opt)

    def get_optimizer(self):
        return self._optimizer_output, self._optimizer_hidden

    def get_gradient_weights(self):
        return self._gradient_W

    def get_weights(self):
        return self.FC_hidden.weights

    def set_weights(self, weights):
        self.FC_hidden.weights = weights

    memorize = property(get_memorize, set_memorize)
    optimizer = property(get_optimizer, set_optimizer)
    gradient_weights = property(get_gradient_weights)
    weights = property(get_weights, set_weights)