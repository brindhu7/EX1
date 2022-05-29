import sys

import numpy as np
from Layers import Base
from Optimization.Optimizers import Sgd


class FullyConnected(Base.BaseLayer):
    def __init__(self,input_size, output_size):
        #calling super constructor
        super().__init__(True)
        #setting inherited member
        #self.trainable = True

        self.input_size = input_size
        self.output_size = output_size
        weights_size = self.input_size + 1
        #initialize the weights
        self.weights = np.random.uniform(0,1,(weights_size,self.output_size))
        self._optimizer = 0

    # getter method
    def get_optimizer(self):
        return self._optimizer

    # setter method
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        # appending bias "1" column to the input_tensor
        ones= np.ones((input_tensor.shape[0],1))
        input_with_bias = np.append(input_tensor,ones, axis = 1)
        #Linear operation on input
        output_tensor = np.dot(input_with_bias,self.weights)
        return output_tensor

    def backward(self,error_tensor):
        error_tensor_prev = np.dot(error_tensor, self.weights[0:self.weights.shape[0]-1, :].T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer:
            Sgd.calculate_update(self.weights, self.gradient_weights)
        return error_tensor_prev








