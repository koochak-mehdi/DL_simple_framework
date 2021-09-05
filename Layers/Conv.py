import numpy as np
from scipy import signal
import copy
from . import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        #print("stride -- ", stride_shape)
        #print("conv_shape -- ", convolution_shape)
        #print("num_kernels -- ", num_kernels)

        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        #print("w_shape -- ", self.weights.shape)

        self.bias = np.random.uniform(0, 1, num_kernels)
        #print("b_shape -- ", self.bias.shape)

        self._weights_optimizer = None
        self._bias_optimizer = None
        self._optimizer = None

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        pass

    def forward(self, input_tensor):
        #print("### forward ###")
        self.input_tensor = input_tensor
        #print("in -- ", input_tensor.shape)


        px, py = self.get_padding_size()
        #print("px -- {}, py -- {}".format(px, py))

        if len(input_tensor.shape) == 3:        # 1D
            b, c, y = input_tensor.shape
            
            self.padded_input = np.pad(input_tensor, ((0,0), (0,0), py))
            #print("padded_input -- ", self.padded_input.shape)

            self.output = np.zeros((input_tensor.shape[0], self.num_kernels, input_tensor.shape[-1]))
            
            for i in range(b):
                for j in range(self.num_kernels):
                    self.output[i, j] = signal.correlate(self.padded_input[i], self.weights[j], mode="valid")
                    self.output += self.bias[j]

            self.upsample_shape = self.output.shape
            self.output = self.output[:,:, ::self.stride_shape[0]]
            #print("out -- ", self.output.shape)
        else:                                   # 2D
            b, c, x, y = input_tensor.shape

            self.padded_input = np.pad(input_tensor, ((0,0), (0,0), px, py))
            #print("padded_input -- ", self.padded_input.shape)

            self.output = np.zeros((b, self.num_kernels, x, y))

            for i in range(b):
                for j in range(self.num_kernels):
                    self.output[i, j] = signal.correlate(self.padded_input[i], self.weights[j], mode="valid")
                    self.output[i, j] += self.bias[j]

            self.upsample_shape = self.output.shape
            self.output = self.output[:,:, ::self.stride_shape[0], ::self.stride_shape[1]]
            #print("out -- ", self.output.shape)


        return self.output

    def backward(self, error_tensor):
        err_output = np.zeros(self.input_tensor.shape)
        px, py = self.get_padding_size()

        err_upsampled = np.zeros(self.upsample_shape)

        new_weights = np.swapaxes(self.weights, axis1=0, axis2=1)

        if len(error_tensor.shape)==3:      # 1D
            err_upsampled[..., ::self.stride_shape[0]] = error_tensor  # upsampling
            padded_err_tensor = np.pad(err_upsampled, ((0, 0), (0, 0), py))

            for i in range(error_tensor.shape[0]):
                for j in range(new_weights.shape[0]):
                    new_kernel = new_weights[j]
                    err_output[i, j] = signal.convolve(padded_err_tensor[i], new_kernel, mode='valid')
            
            self._gradient_weights = np.zeros_like(self.weights)
            padded_input = np.pad(self.input_tensor, ((0,0), (0,0), py))
            for i in range(self.input_tensor.shape[0]):
                for j in range(new_weights.shape[1]):
                    for k in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j, k] += signal.correlate(padded_input[i, k], err_upsampled[i, j], mode='valid')
            
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))
        else:                               # 2D
            err_upsampled[:,:,::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
            padded_err_tensor = np.pad(err_upsampled, ((0,0), (0,0), px, py))

            for i in range(error_tensor.shape[0]):
                for j in range(new_weights.shape[0]):
                    new_kernel = np.flip(new_weights[j], axis=0)
                    err_output[i, j] = signal.convolve(padded_err_tensor[i], new_kernel, mode='valid')
            
            self._gradient_weights = np.zeros_like(self.weights)
            padded_input = np.pad(self.input_tensor, ((0,0), (0,0), px, py))
            for i in range(self.input_tensor.shape[0]):
                for j in range(error_tensor.shape[1]):
                    for k in range(self.input_tensor.shape[1]):
                        self._gradient_weights[j, k] += signal.correlate(padded_input[i,k], err_upsampled[i, j], mode='valid')
            
            self._gradient_bias = np.sum(error_tensor, axis=(0,2,3))

        if self._bias_optimizer is not None:
            opt = self._bias_optimizer
            self.bias = opt.calculate_update(self.bias, self._gradient_bias)
        
        if self._weights_optimizer is not None:
            opt = self._weights_optimizer
            self.weights = opt.calculate_update(self.weights, self._gradient_weights)

        return err_output

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,
                                                        np.prod(self.convolution_shape),
                                                        self.num_kernels*np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape,
                                                0,
                                                0)
        pass

    def get_padding_size(self):
        if len(self.convolution_shape) == 3:    # 2D
            px_1 = self.convolution_shape[1]//2
            px_2 = px_1
            if self.convolution_shape[1] % 2 == 0:
                px_1 -= 1
            
            py_1 = self.convolution_shape[2]//2
            py_2 = py_1
            if self.convolution_shape[2] % 2 == 0:
                py_1 -= 1
        else:                                   # 1D
            px_1 = 0
            px_2 = 0
            py_1 = self.convolution_shape[1]//2
            py_2 = py_1
            if self.convolution_shape[1] % 2 == 0:
                py_1 -= 1
        return (px_1, px_2), (py_1, py_2)

    def set_optimizer(self, opt):
        self._optimizer = opt
        self._weights_optimizer = copy.deepcopy(opt)
        self._bias_optimizer = copy.deepcopy(opt)
        
    def get_optimizer(self):
        return self._optimizer

    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias

    optimizer = property(get_optimizer, set_optimizer)
    gradient_weights = property(get_gradient_weights)
    gradient_bias = property(get_gradient_bias)
