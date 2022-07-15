import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase:
            x = input_tensor
        else:
            t = np.linspace(0, 1, input_tensor.shape[-1])
            t[0] = 0.001

            t[t > self.probability] = 0
            t[t > 0] = 1
            np.random.shuffle(t)
            self.dropped = t
            x = input_tensor*t
            x = np.round(x/self.probability)
        return x

    def backward(self, error_tensor):
        return error_tensor*self.dropped /self.probability