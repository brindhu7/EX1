import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,prediction_tensor,label_tensor):
        epsilon = np.finfo(float).eps
        error_matrix = -np.sum(label_tensor * np.log(prediction_tensor + epsilon))
        return error_matrix

    def backward(self,label_tensor):
        return error_tensor
