import numpy as np
class SoftMax:
    def __init__(self):
        pass

    def forward(self,input_tensor):
        #find maximum value for each row in batch

        numerator = np.exp(input_tensor - np.max(input_tensor,axis=1, keepdims=True))
        output_tensor = numerator/np.sum(numerator)
        return output_tensor

    def backward(self,error_tensor):
        return error_tensor
