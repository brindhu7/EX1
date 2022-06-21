import numpy as np
from Layers.Base import BaseLayer
from scipy.signal import correlate2d, correlate
from scipy.ndimage import convolve1d, convolve

class Conv(BaseLayer):
    def __init__(self,stride_shape, convolution_shape, num_kernels):
        # calling super constructor - base class
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # setting inherited member
        self.trainable = True
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size=num_kernels)
        self.gradient_weights = None
        self.gradient_bias = None
        self.input_tensor = None
        self.optimizer = None

    def forward_2D(self, input_tensor):
        batch_size = input_tensor.shape[0]
        width = input_tensor.shape[3]
        height = input_tensor.shape[2]
        stride_y = self.stride_shape[1]
        stride_x = self.stride_shape[0]
        num_channels = self.convolution_shape[0]

        output_tensor = np.zeros((batch_size, self.num_kernels, int(np.ceil(height/stride_x)), int(np.ceil(width/stride_y))))

        for n_b in range(batch_size):
            for n_k in range(self.num_kernels):
                result_per_ch = []
                for n_c in range(num_channels):
                    result_per_ch.append(correlate2d(input_tensor[n_b][n_c], self.weights[n_k][n_c], mode='same', boundary='fill'))
                temp = np.sum(np.array(result_per_ch), axis=0)
                temp = temp[::stride_x, ::stride_y] + self.bias[n_k]
                output_tensor[n_b][n_k] = temp
        return output_tensor

    def forward_1D(self, input_tensor):
        batch_size = input_tensor.shape[0]
        width = input_tensor.shape[-1]
        stride_y = self.stride_shape[0]
        num_channels = self.convolution_shape[0]

        output_tensor = np.zeros((batch_size, self.num_kernels, int(np.ceil(width / stride_y))))

        for n_b in range(batch_size):
            for n_k in range(self.num_kernels):
                result = []
                for n_c in range(num_channels):
                    result.append(correlate(input_tensor[n_b][n_c], self.weights[n_k][n_c], mode='same'))
                temp = np.sum(np.array(result), axis=0)
                output_tensor[n_b][n_k] = temp[::stride_y] + self.bias[n_k]
        return output_tensor

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 4:
            output_tensor = self.forward_2D(input_tensor)
        else:
            output_tensor = self.forward_1D(input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        if len(error_tensor.shape)==4:
            error_tensor_out = self.backward_2D(error_tensor)
        else:
            error_tensor_out = self.backward_1D(error_tensor)

        return error_tensor_out

    def backward_2D(self, error_tensor):
        batch_size = self.input_tensor.shape[0]
        width = self.input_tensor.shape[3]
        height = self.input_tensor.shape[2]
        stride_y = self.stride_shape[1]
        stride_x = self.stride_shape[0]
        num_channels = self.convolution_shape[0]

        error_tensor_rshpd = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        error_tensor_rshpd[:, :, ::stride_x, ::stride_y] = error_tensor

        Kernel_rshpd = np.zeros((num_channels, self.num_kernels, *self.convolution_shape[1:]))

        for n_c in range(num_channels):
            for n_k in range(self.num_kernels):
                Kernel_rshpd[n_c][n_k] = self.weights[n_k][n_c]

        error_tensor_out = np.zeros((batch_size, num_channels, height, width))

        for n_b in range(batch_size):
            for n_c in range(num_channels):
                result = []
                for n_k in range(self.num_kernels):
                    result.append(convolve(error_tensor_rshpd[n_b][n_k], Kernel_rshpd[n_c][n_k], mode='constant', cval=0))
                error_tensor_out[n_b][n_c] = np.sum(np.array(result), axis=0)

        self.gradient_bias = np.zeros(self.bias.shape)
        for n_k in range(self.num_kernels):
            self.gradient_bias[n_k] = np.sum(error_tensor[:, n_k, :, :])


        delta_weights = np.zeros(self.weights.shape)
        conv_shape = np.array(self.convolution_shape)-1
        _,lx,ly = np.int32(np.floor(conv_shape/2))
        _,rx,ry = np.int32(np.ceil(conv_shape/2))
        padded_input_tensor = np.pad(self.input_tensor, ((0,0),(0,0),(lx,rx),(ly,ry)))

        for n_b in range(batch_size):
            for n_k in range(self.num_kernels):
                result = []
                for n_c in range(num_channels):
                    result.append(correlate2d(padded_input_tensor[n_b][n_c],error_tensor_rshpd[n_b][n_k],mode='valid'))
                delta_weights[n_k] += np.array(result)
        self.gradient_weights = delta_weights

        if self.optimizer is not None:
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_tensor_out

    def backward_1D(self, error_tensor):
        batch_size = self.input_tensor.shape[0]
        width = self.input_tensor.shape[2]
        stride_y = self.stride_shape[0]
        num_channels = self.convolution_shape[0]

        error_tensor_rshpd = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        error_tensor_rshpd[:, :, ::stride_y] = error_tensor

        Kernel_rshpd = np.zeros((num_channels, self.num_kernels, self.convolution_shape[-1]))

        for n_c in range(num_channels):
            for n_k in range(self.num_kernels):
                Kernel_rshpd[n_c][n_k] = self.weights[n_k][n_c]

        error_tensor_out = np.zeros((batch_size, num_channels, width))

        for n_b in range(batch_size):
            for n_c in range(num_channels):
                result = []
                for n_k in range(self.num_kernels):
                    result.append(convolve1d(error_tensor_rshpd[n_b][n_k], Kernel_rshpd[n_c][n_k], mode='constant', cval=0))
                error_tensor_out[n_b][n_c] = np.sum(np.array(result), axis=0)

        self.gradient_bias = np.zeros(self.bias.shape)
        for n_k in range(self.num_kernels):
            self.gradient_bias[n_k] = np.sum(error_tensor[:, n_k, :])

        delta_weights = np.zeros_like(self.weights)
        conv_shape = np.array(self.convolution_shape)-1
        _,l= np.int32(np.floor(conv_shape/2))
        _,r = np.int32(np.ceil(conv_shape/2))
        padded_input_tensor = np.pad(self.input_tensor,((0,0),(0,0),(l,r)))

        for n_b in range(batch_size):
            for n_k in range(self.num_kernels):
                result = []
                for n_c in range(num_channels):
                    result.append(correlate(padded_input_tensor[n_b][n_c],error_tensor_rshpd[n_b][n_k],mode='valid'))
                delta_weights[n_k] += np.array(result)
        self.gradient_weights = delta_weights

        if self.optimizer is not None:
            self.optimizer.calculate_update(self.bias,self.gradient_bias)
            self.optimizer.calculate_update(self.weights,self.gradient_weights)

        return error_tensor_out

    def initialize(self, weights_initializer, bias_initializer):
        weight_shape = (self.num_kernels, * self.convolution_shape)
        fan_out_shape = self.num_kernels * np.product(self.convolution_shape[1:])
        fan_in_shape = np.prod(self.convolution_shape)
        self.weights = weights_initializer.initialize(weight_shape,
                                                      fan_in_shape, fan_out_shape)
        self.bias = bias_initializer.initialize(self.num_kernels, 1, self.num_kernels)




