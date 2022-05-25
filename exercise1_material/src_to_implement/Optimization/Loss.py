import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,prediction_tensor,label_tensor):
        #getting the index of the "1" in one hot encoded label_tensor
        index = np.where(label_tensor==1)
        #get the elements of these index in softmax output tensor(prediction_tensor)
        self.prediction_tensor = prediction_tensor[index]
        epsilon = np.finfo(float).eps
        loss_error_matrix = np.sum(-np.log(self.prediction_tensor + epsilon))
        return loss_error_matrix

    def backward(self,label_tensor):
        return error_tensor
