
class ReLU:
    def __init__(self):
        pass

    def forward(self,input_tensor):
        input_tensor[input_tensor <= 0] = 0
        output_tensor = input_tensor
        return output_tensor

    def backward(self,error_tensor):
        return error_tensor
