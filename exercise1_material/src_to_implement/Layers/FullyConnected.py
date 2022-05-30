
import numpy as np
from exercise1_material.src_to_implement.Layers.Base import BaseLayer
import sys
sys.path.append('C:\\Users\\brind\\PycharmProjects\\EX1\\exercise1_material\\src_to_implement\\Optimization')


class FullyConnected(BaseLayer):
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
        self._optimizer=None



    def forward(self,input_tensor):
        # appending bias "1" column to the input_tensor
        ones= np.ones((input_tensor.shape[0],1))
        self.input_with_bias = np.append(input_tensor,ones, axis = 1)
        #Linear operation on input
        output_tensor = np.dot(self.input_with_bias,self.weights)
        return output_tensor

    #getter property
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,value):
        self._optimizer = value



    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        # error_gradient w.r.t input without bias, En-1 always shape of input
        error_tensor_Prev = np.dot(self.error_tensor, np.transpose(self.weights))[:, :-1]

        if self._optimizer:
            self.gradient__weights = self.gradient_weights
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient__weights)

        return error_tensor_Prev

    @property
    def gradient_weights(self):
        weight_gradients = np.dot(np.transpose(self.input_with_bias), self.error_tensor)
        return weight_gradients


