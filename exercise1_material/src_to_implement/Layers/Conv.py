import numpy as np
from Layers.Base import BaseLayer
from scipy.ndimage.filters import correlate, convolve
from scipy.signal import correlate as cr
from scipy.signal import convolve as cv


class Conv(BaseLayer):
    def __init__(self,stride_shape, convolution_shape,num_kernels):
        # calling super constructor - base class
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # setting inherited member
        self.trainable = True
        try:
            self.weights = np.random.randn(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
        except Exception:
            self.weights = np.random.randn(num_kernels, convolution_shape[0], convolution_shape[1])
        self.bias = np.zeros(num_kernels)
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None

    def forward(self, input_tensor):
        # input_tensor shape: b c y x (D x C x H x W)
        # self.weights shape: NF x C x HF x HW
        self.optimizer = input_tensor
        if len(input_tensor.shape) == 4:
            if (self.stride_shape[0] != 1 and len(self.stride_shape) == 1)  or \
                    (self.stride_shape[0] != 1 and self.stride_shape[1] != 1 and  len(self.stride_shape) == 2):
                if len(input_tensor.shape) == 4:
                    size_y = int(np.ceil(input_tensor.shape[-1] / self.stride_shape[1]))
                    size_x = int(np.ceil(input_tensor.shape[-2] / self.stride_shape[0]))
                    input = np.zeros(
                        [input_tensor.shape[0], input_tensor.shape[1], self.stride_shape[0] * size_x,
                         self.stride_shape[1] * size_y])
                    input[0:input_tensor.shape[0], 0:input_tensor.shape[1], 0:input_tensor.shape[2], 0:input_tensor.shape[3]] = input_tensor
                    output_tensor = []
                    for i in range(self.num_kernels):
                        temp_x = []
                        for j in range(size_y):
                            temp_y = []
                            for k in range(size_x):
                                left = j*self.stride_shape[1]
                                right = left + self.stride_shape[1]
                                top = k*self.stride_shape[0]
                                bottom = top + self.stride_shape[0]
                                temp_y.append(np.sum(np.sum(np.sum(np.array(correlate(input[:, :, top:bottom, left:right], self.weights[i, :, :, :][np.newaxis, :, :, :], mode='constant') + self.bias[i]), 3), 2), 1))
                                #temp_y.append(correlate(input[:, :, top:bottom, left:right], self.weights[i, :, :, :][np.newaxis, :, :, :],
                                #                 mode='constant') + self.bias[i])
                            temp_x.append(temp_y)
                        output_tensor.append(temp_x)
                    output_tensor = np.transpose(np.array(output_tensor), [3, 0, 2, 1])
                    print("Hi")
            else:
                output_tensor = np.zeros([input_tensor.shape[0], self.num_kernels, input_tensor.shape[2], input_tensor.shape[3]])
                for i in range(self.num_kernels):
                    temp = correlate(input_tensor, self.weights[i, :, :, :][np.newaxis, :, :, :], mode='constant') + self.bias[i]
                    if len(temp.shape) == 4:
                        temp = np.sum(temp, 1)
                    output_tensor[:, i, :, :] = temp
        else:
            if (self.stride_shape[0] != 1) :
                size_y = int(np.ceil(input_tensor.shape[-1] / self.stride_shape[0]))
                input = np.zeros(
                    [input_tensor.shape[0], input_tensor.shape[1], self.stride_shape[0] * size_y])
                input[0:input_tensor.shape[0], 0:input_tensor.shape[1], 0:input_tensor.shape[2]] = input_tensor
                output_tensor = []
                for i in range(self.num_kernels):
                    temp_x = []
                    for j in range(size_y):
                        left = j * self.stride_shape[0]
                        right = left + self.stride_shape[0]
                        temp_x.append(np.sum(np.sum(np.array(correlate(input[:, :, left:right], self.weights[i, :, :][np.newaxis, :, :],
                                      mode='constant') + self.bias[i]), 2), 1))
                    output_tensor.append(temp_x)
                output_tensor = np.transpose(np.array(output_tensor), [2, 0, 1])
                print("Hi")
            else:
                output_tensor = np.zeros(
                    [input_tensor.shape[0], self.num_kernels, input_tensor.shape[1], input_tensor.shape[2]])
                for i in range(self.num_kernels):
                    output_tensor[:, i, :, :] = correlate(input_tensor, self.weights[i, :, :][np.newaxis, :, :],
                                                          mode='constant') + self.bias[i]

        self.out = output_tensor
        return output_tensor

    def backward(self, error_tensor):

        if self.optimizer is not None:
            if len(self.optimizer.shape)==4:
                #error_tensor_out = np.zeros([error_tensor.shape[0], error_tensor.shape[1]-1, error_tensor.shape[2], error_tensor.shape[3]])
                error_tensor_out = []
                for i in range(np.size(self.optimizer, 0)): # batches
                    error_tensor_out_ch = []
                    for j in range(np.size(self.optimizer, 1)): # channels
                        error_tensor_out_filters = []
                        for k in range(self.num_kernels): # filters
                            error_tensor_out_filters.append(convolve(error_tensor[i, k, :, :], self.weights[k, j, :, :],
                                                          mode='constant'))
                        error_tensor_out_ch.append(np.sum(error_tensor_out_filters, 0))
                    error_tensor_out.append(error_tensor_out_ch)
            else:
                error_tensor_out = []
                for i in range(np.size(self.optimizer, 0)):  # batches
                    error_tensor_out_ch = []
                    for k in range(self.num_kernels):  # filters
                        error_tensor_out_ch.append(convolve(error_tensor[i, k, :, :], self.weights[k, :, :],
                                                                 mode='constant'))
                    error_tensor_out.append(np.sum(error_tensor_out_ch, 0))

            self.gradient_weights = []
            for i in range(np.size(self.optimizer, 0)):  # batches

                mid_ind = (self.optimizer.shape[-1] + error_tensor.shape[-1] - 2)//2
                if len(self.optimizer.shape) == 4:
                    gd = np.flip(cr(self.optimizer[i, :, :, :][np.newaxis, :, :, :], error_tensor[i, :, :, :][:, np.newaxis, :, :]), 0)[:, :, mid_ind-1:mid_ind+2, mid_ind-1:mid_ind+2]
                else:
                    gd = np.flip(cr(self.optimizer[i, :, :][np.newaxis, :, :],
                                    error_tensor[i, :, :, :]), 0)[:, :, mid_ind - 1:mid_ind + 2]
                self.gradient_weights.append(gd)

            self.gradient_bias = []
            for i in range(np.size(error_tensor, 1)):  # channels
                  self.gradient_bias.append(np.sum(error_tensor[:, i, :, :]))

        return np.array(error_tensor_out)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer
        self.bias = bias_initializer




