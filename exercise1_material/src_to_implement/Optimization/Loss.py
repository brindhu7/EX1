import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = np.finfo(float).eps
        pass

    def forward(self, prediction_tensor, label_tensor):
        # getting the prediction tensor for error tensor calculation in backward path
        self.prediction_tensor = prediction_tensor

        ##getting the index of the "1" in one hot encoded label_tensor
        index = np.where(label_tensor == 1)
        # get the elements of these index in softmax output tensor(prediction_tensor)
        act_layer_op = prediction_tensor[index]
        # Loss for all the entire batch input
        loss = np.sum(-np.log(act_layer_op + self.epsilon))

        #loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon))
        return loss

    def backward(self, label_tensor):
        error_tensor = -label_tensor / (self.prediction_tensor)
        return error_tensor




