import Optimization
from . import Base
import numpy as np
import copy

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable=True
        self.stride_shape=stride_shape
        self.convolution_shape=convolution_shape
        self.num_kernels=num_kernels
        #print(convolution_shape)
        #print(num_kernels)
        self.weights= np.random.uniform(0,1,(num_kernels, *convolution_shape))
        self.bias=np.random.uniform(0,1,num_kernels)
        self.bias_optimizer=None
        self.weights_optimizer=None
        self._gradient_weights= None
        self._gradient_bias= None

    def forward(self, input_tensor):
        self.input_tensor=input_tensor
        #print( input_tensor.shape)
        #print (len(input_tensor.shape))
        px,py= self.calculate_padding()

        if (len(input_tensor.shape)==3):        # 1D
            b,c,y= input_tensor.shape
            pad_input= np.pad(input_tensor,((0,0),(0,0),py))
            self.output=np.zeros((b,self.num_kernels,y))
            
            for i in range(b):
                for j in range(self.num_kernels):
                    self.output[i,j]= signal.correlate(pad_input[i], self.weights[j],mode='valid')
                    self.output[i,j]+=self.bias[j]
            
            self.up_shape= self.output.shape
            self.output= self.output[:,:,::self.stride_shape[0]]
        else:                                   # 2D
            b,c,x,y=input_tensor.shape
            pad_input= np.pad(input_tensor,((0,0),(0,0),px,py))
            #print(pad_input.shape)
            self.output= np.zeros((b,self.num_kernels,x,y))
            for i in range(b):
                for j in range(self.num_kernels):
                    self.output[i,j]= signal.correlate(pad_input[i], self.weights[j],mode='valid')
                    self.output[i,j]+=self.bias[j]
            
            self.up_shape= self.output.shape
            self.output= self.output[:,:, ::self.stride_shape[0], :: self.stride_shape[1]]
                                    # [b,c, start stop step[axis]]
            pass
        
        return self.output

    def backward(self, error_tensor):
        #print (self.up_shape)
        #print (self.input_tensor.shape)
        back_out= np.zeros_like(self.input_tensor)
        px,py= self.calculate_padding()
        err_plane= np.zeros(self.up_shape)
        new_weights= np.swapaxes(self.weights, axis1=0, axis2=1)

        if (len(error_tensor.shape)==3):                    #1D
            err_plane[:,:,::self.stride_shape[0]]=error_tensor
            pad_err_tensor= np.pad(err_plane,((0,0), (0,0), py))
            for i in range(error_tensor.shape[0]):
                for j in range(new_weights.shape[0]):
                    new_kernel= new_weights[j]
                    back_out[i,j]= signal.convolve(pad_err_tensor[i], new_kernel, mode='valid')
            
            self._gradient_weights=np.zeros_like(self.weights)
            padded_input= np.pad(self.input_tensor, ((0,0), (0,0), py))
            for i in range(self.input_tensor.shape[0]):
                for j in range(error_tensor.shape[1]):
                    for k in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j,k]+=signal.correlate(padded_input[i,k], err_plane[i,j], mode="valid")
            
            self._gradient_bias=np.sum(error_tensor, axis=(0,2))

                                              
        else:                                         #2D       
            err_plane[:,:,::self.stride_shape[0],::self.stride_shape[1]]=error_tensor
            pad_err_tensor= np.pad(err_plane,((0,0),(0,0), px, py))
            for i in range(error_tensor.shape[0]):
                for j in range(new_weights.shape[0]):
                    new_kernel = np.flip(new_weights[j], axis=0)
                    back_out[i,j]= signal.convolve(pad_err_tensor[i], new_kernel, mode='valid') 

            self._gradient_weights= np.zeros_like(self.weights)
            padded_input= np.pad(self.input_tensor,((0,0), (0,0), px, py))
            for i in range(self.input_tensor.shape[0]):
                for j in range(error_tensor.shape[1]):
                    for k in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j, k ]+= signal.correlate(padded_input[i,k] ,  err_plane[i,j], mode='valid')

            self._gradient_bias= np.sum( error_tensor, axis=(0,2,3))
        
        if self.bias_optimizer is not None:
            opt= self.bias_optimizer
            self.bias= opt.calculate_update(self.bias, self._gradient_bias)

        if self.weights_optimizer is not None:
            opt= self.weights_optimizer
            self.weights= opt.calculate_update(self.weights, self._gradient_weights)


        return back_out

    def set_optimizer(self, optimizer):
        self.weights_optimizer=optimizer
        self.bias_optimizer=copy.deepcopy(optimizer)

    def get_optimizer(self):
        return self.weights_optimizer, self.bias_optimizer
    
    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias

    optimizer = property(get_optimizer,set_optimizer)
    gradient_weights= property(get_gradient_weights)
    gradient_bias= property(get_gradient_bias)


    def calculate_padding(self):
        if (len(self.input_tensor.shape)==3):
            px_1=0
            px_2=0
            #c,y
            py_1= self.convolution_shape[1]//2
            py_2= py_1
            if self.convolution_shape[1]%2==0:
                py_2-=1
            pass
        else:
            #c,x,y
            px_1= self.convolution_shape[1]//2
            px_2= px_1
            if self.convolution_shape[1]%2==0:
                px_2-=1

            py_1= self.convolution_shape[2]//2
            py_2= py_1
            if self.convolution_shape[2]%2==0:
                py_2-=1

        return (px_1,px_2),(py_1,py_2)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights=weights_initializer.initialize(self.weights.shape , np.prod(self.convolution_shape), self.num_kernels*np.prod(self.convolution_shape[1:]) )
        self.bias=bias_initializer.initialize(self.bias.shape, 0, 0)