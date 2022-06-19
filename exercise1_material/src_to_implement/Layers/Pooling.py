import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        print("Hi")
        count_k = 0
        count_l = 0
        A = np.zeros([input_tensor.shape[0], input_tensor.shape[1], self.pooling_shape[0], self.pooling_shape[1]])
        for i in range(input_tensor.shape[0]):
           for j in range(input_tensor.shape[1]):
               count_k = 0
               for k in range(0, input_tensor.shape[2], self.stride_shape[0]):
                   count_k = count_k + 1
                   count_l = 0
                   for l in range(0, input_tensor.shape[3], self.stride_shape[1]):
                       count_l = count_l + 1
                       n_left = l
                       n_right = l + self.stride_shape[1]
                       n_top = k
                       n_down = k + self.stride_shape[0]
                       A[i, j, count_k, count_l] = np.max(input_tensor[i, j, n_left:n_right, n_top:n_down])
        return A
        #pass

    def backward(self, error_tensor):
        print("Hi")
        pass