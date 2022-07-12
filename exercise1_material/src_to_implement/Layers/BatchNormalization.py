from exercise1_material.src_to_implement.Layers.Base import BaseLayer
import numpy as np
from .Helpers import compute_bn_gradients
class BatchNormalization(BaseLayer):
    def __init__(self,channels):
        self.channels = channels
        super().__init__()
        self.trainable = True
        self.epsilon = np.finfo(np.float).eps
        self.alpha = 0.8
        self.mov_avg_mu = 0
        self.mov_avg_var = 0
        self.mu=0
        self.variance=0
        self.bias = np.zeros(self.channels)  # bias
        self.weights = np.ones(self.channels)  # weights
        self.imageflag = False
        self.first_batch = True
        self.normalized_input = None
        self._optimizer = None

    def initialize(self):
        self.bias = np.zeros(self.channels) #bias
        self.weights = np.ones(self.channels) #weights

    def forward(self,input_tensor):
        # reformat the input tensor for images #CNN
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
            self.imageflag = True


        self.input_tensor = input_tensor

        #in training phase the mean and variance are calculated batch wise for each feature(each column- axis= 0)
        if not self.testing_phase:
            self.mu = np.mean(self.input_tensor,axis = 0)
            self.variance = np.var(self.input_tensor,axis = 0)
            #storing the moving averages of mean and variance
            self.mov_avg_mu = self.alpha * self.mov_avg_mu + (1 - self.alpha) * self.mu
            self.mov_avg_var = self.alpha * self.mov_avg_var + (1 - self.alpha) * self.variance
        else:
            self.mu = self.mov_avg_mu
            self.variance = self.mov_avg_var


        self.normalized_input = (self.input_tensor - self.mu ) / np.sqrt(self.variance + self.epsilon)
        #scaling and shifiting is done with gamma and beta(weights and bias)
        output_tensor = self.weights * self.normalized_input + self.bias
        if self.imageflag:
            output_tensor = self.reformat(output_tensor)
            self.imageflag = False

        #initializing the moving average with first batch mean and variance
        if self.first_batch:
            self.mov_avg_mu = self.mu
            self.mov_avg_var = self.variance
            self.first_batch = False

        return output_tensor

    def backward(self,error_tensor):
        if len(error_tensor.shape) == 4:
            error_tensor=self.reformat(error_tensor)
            self.imageflag=True

        self.error_tensor = error_tensor

        self.gradient_weights = np.sum(self.normalized_input * self.error_tensor,axis=0)
        self.gradient_bias = np.sum(self.error_tensor,axis=0)

        gradient_wrt_inputs = compute_bn_gradients(self.error_tensor,self.input_tensor,self.weights,self.mu,self.variance)

        if self.imageflag:
            gradient_wrt_inputs = self.reformat(gradient_wrt_inputs)

        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)


        return gradient_wrt_inputs


    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            B, H, M, N = tensor.shape
            self.tensor_shape = tensor.shape
            reshaped_tensor = np.reshape(tensor, (B, H, M * N))
            transposed = np.transpose(reshaped_tensor, (0, 2, 1))
            refactored_tensor = np.reshape(transposed, (B * M * N, H))
        elif (len(tensor.shape)) == 2:
            b, h, m, n = self.tensor_shape
            reshaped = np.reshape(tensor, (b, m * n, h))
            transposed = np.transpose(reshaped, (0, 2, 1))
            refactored_tensor = np.reshape(transposed, (b, h, m, n))

        return refactored_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value


