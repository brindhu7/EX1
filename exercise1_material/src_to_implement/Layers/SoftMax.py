import numpy as np
from Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        #find maximum value for each row in batch
        numerator = np.exp(input_tensor - np.max(input_tensor,axis=1, keepdims=True))
        self.output_tensor = numerator/np.sum(numerator)
        return self.output_tensor

    def backward(self,error_tensor):
        error_tensor_prev = self.output_tensor * (error_tensor - np.sum(self.output_tensor*error_tensor, 1)[:, np.newaxis])
        return error_tensor_prev
