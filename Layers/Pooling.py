from . import Base
import numpy as np

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.sx, self.sy = stride_shape
        self.px, self.py = pooling_shape

    def forward(self, input_tensor):
        #print('in -- ', input_tensor.shape)
        #print('pool -- ', (self.px, self.py))
        self.input_tensor = input_tensor
        batch,channel,ht,wt = input_tensor.shape
        output = np.zeros((batch, channel, 
                    int((ht-self.px)/self.sx)+1, 
                    int((wt-self.py)/self.sy)+1))
        
        for b in range(batch):
            for c in range(channel):
                x = 0
                for k in range(output.shape[2]):
                    y = 0
                    for h in range(output.shape[3]):
                        output[b,c,k,h] = np.max(input_tensor[b,c,x:x+self.px, y:y+self.py])
                        y += self.sy
                    x += self.sx

        return output
    
    def backward(self, error_tensor):
        batch, channel, ht, wt = error_tensor.shape
        err_output = np.zeros_like(self.input_tensor)

        for b in range(batch):
            for c in range(channel):
                x = 0
                for i in range(ht):
                    y = 0
                    for j in range(wt):
                        pooled = self.input_tensor[b,c, x:x+self.px, y:y+self.py]

                        mask = [pooled == np.max(pooled)]
                        err_output[b,c,
                            x:x+self.px,
                            y:y+self.py][mask] += error_tensor[b,c,i,j]
                        y += self.sy 
                    x += self.sx
        
        return err_output