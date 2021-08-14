import numpy as np
from numpy.core.fromnumeric import shape 
from . import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape , pooling_shape):
        super().__init__()

        self.px, self.py= pooling_shape
        self.sx, self.sy= stride_shape

    def forward(self, input_tensor):
        self.input_tensor= input_tensor
        dx= int((input_tensor.shape[2]-self.px)/self.sx)+1
        dy= int((input_tensor.shape[3]-self.py)/ self.sy)+1
        output= np.zeros((input_tensor.shape[0], input_tensor.shape[1], dx, dy))
       

        for b in range(self.input_tensor.shape[0]):
            for c in range(self.input_tensor.shape[1]):
                x0 = 0
                for i in range(output.shape[2]):
                    y0 = 0
                    for j in range(output.shape[3]):
                        output[b, c, i, j] = np.max(input_tensor[b, c, x0:x0+self.px, y0:y0+self.py])
                        y0+=self.sy
                    x0+=self.sx


        return output


    def backward(self, error_tensor):
        err_output= np.zeros_like(self.input_tensor)  # create plane 

        for b in range(error_tensor.shape[0]):
            for c in range (error_tensor.shape[1]):
                x0 = 0
                for i in range( error_tensor.shape[2]):
                    y0 = 0
                    for j in range( error_tensor.shape[3]):
                        selcted_window = self.input_tensor[b, c , x0:x0+self.px, y0:y0+self.py]
                        mask= selcted_window==np.max(selcted_window)
                        err_output[b,c, x0:x0+self.px, y0:y0+self.py][mask]+= error_tensor[b,c,i,j]
                        y0+=self.sy
                    x0+=self.sx

        return err_output