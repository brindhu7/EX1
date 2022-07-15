import numpy as np
from exercise1_material.src_to_implement.Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None
        pass

    def forward(self,input_tensor):
        self.output = np.tanh(input_tensor)
        return self.output
    def backward(self,error_tensor):
        self.derivative = 1 - self.output ** 2
        return self.derivative*error_tensor