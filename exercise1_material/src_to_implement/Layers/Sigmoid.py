import numpy as np
from exercise1_material.src_to_implement.Layers.Base import BaseLayer
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input_tensor):
        self.output = 1/(1 + np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        self.derivative = self.output*(1 - self.output)
        return self.derivative * error_tensor
