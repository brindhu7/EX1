class Flatten:
    def __init__(self):
        pass
    def forward(self,input_tensor):
        self.input_tensor = input_tensor.flatten()
        return self.input_tensor
    def backward(self,error_tensor):
        self.error_tensor = error_tensor.flatten()
        return self.error_tensor
