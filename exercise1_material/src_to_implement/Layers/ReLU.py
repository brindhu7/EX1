from Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self,input_tensor):
        self.mask = (input_tensor <= 0)
        input_tensor[self.mask] = 0
        return input_tensor

    def backward(self,error_tensor):
        error_tensor[self.mask] = 0
        return error_tensor

