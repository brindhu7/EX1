import numpy as np
from exercise1_material.src_to_implement.Layers.Base import BaseLayer



class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        super().__init__()


    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        batch,channel,height,width = input_tensor.shape
        out_h = (height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        out_w = (width - self.pooling_shape[1]) // self.stride_shape[1] + 1
        output = np.zeros(shape=(batch,channel,out_h,out_w))

        for i in range(batch):
            for c in range(channel):
                for h in range(out_h):
                    for w in range(out_w):
                        output[i,c,h,w] = np.max(self.input_tensor[i,c,
                                                 h*self.stride_shape[0]:h*self.stride_shape[0]+self.pooling_shape[0],
                                                 w*self.stride_shape[1]:w*self.stride_shape[1]+self.pooling_shape[1]])

        return output



    def backward(self,error_tensor):
        B,C, H, W = error_tensor.shape
        dx = np.zeros(self.input_tensor.shape)

        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        index_h,index_w=np.where(np.max(
                            self.input_tensor[b,c,h*self.stride_shape[0]:h*self.stride_shape[0]+self.pooling_shape[0],
                            w*self.stride_shape[1]:w*self.stride_shape[1]+self.pooling_shape[1]]) ==
                                                 self.input_tensor[b,c,h*self.stride_shape[0]:h*self.stride_shape[0]+
                                                                                              self.pooling_shape[0],
                                                 w*self.stride_shape[1]:w*self.stride_shape[1]+self.pooling_shape[1]])
                        index_h,index_w = index_h[0],index_w[0]
                        dx[b,c,h*self.stride_shape[0]:h*self.stride_shape[0]+self.pooling_shape[0],
                        w*self.stride_shape[1]:w*self.stride_shape[1]+self.pooling_shape[1]][index_h,index_w] = \
                            error_tensor[b,c,h,w]
        return dx


