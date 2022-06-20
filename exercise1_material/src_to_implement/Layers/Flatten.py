from Layers.Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.input_tensor_shape = input_tensor.shape
        self.input_tensor = np.reshape(input_tensor, [input_tensor.shape[0], np.prod(input_tensor.shape[1::])])
        return self.input_tensor

    def backward(self,error_tensor):
        self.error_tensor = np.reshape(error_tensor, self.input_tensor_shape)
        return self.error_tensor
