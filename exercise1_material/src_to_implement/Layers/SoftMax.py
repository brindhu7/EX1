import numpy as np
from exercise1_material.src_to_implement.Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        #find maximum value for each sample(row wise) in batch
        #keepdims = True to avoid broadcast error
        numerator = np.exp(input_tensor - np.max(input_tensor,axis=1, keepdims=True))
        self.output_tensor  = numerator/np.sum(numerator,axis = 1,keepdims=True)
        #now input_tensor changes to tensor with probbility values b/w 0 to 1
        return self.output_tensor

    def backward(self, error_tensor):
        error = np.sum(self.output_tensor * error_tensor, axis=1, keepdims=True)
        error_prev = self.output_tensor * (error_tensor - error)
        return error_prev
