import sys

import numpy as np
from Layers import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self,input_size, output_size):
        #calling super constructor
        super().__init__()
        #setting inherited member
        self.trainable = True

        self.input_size = input_size
        self.output_size = output_size
        weights_size = self.input_size + 1
        #initialize the weights
        self.weights = np.random.uniform(0,1,(weights_size,self.output_size))

    def forward(self,input_tensor):
        # appending bias "1" column to the input_tensor
        ones= np.ones((input_tensor.shape[0],1))
        input_with_bias = np.append(input_tensor,ones, axis = 1)
        #Linear operation on input
        output_tensor = np.dot(input_with_bias,self.weights)
        return output_tensor

    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        error_tensor_Prev = np.dot((np.transpose(self.weights),self.error_tensor))





