import numpy as np
class Constant:
    def __init__(self,x):
        self.x = x

    def initialize(self,weights_shape,fan_in,fan_out):
        init_weight_tensor = np.ones(fan_in,fan_out) * self.x
        return init_weight_tensor


class UniformRandom:

    def initialize(self,weights_shape,fan_in,fan_out):
        init_weight_tensor = np.random.uniform(0,1,size= (fan_in,fan_out))
        return init_weight_tensor

class Xavier:

    def initialize(self,weights_shape,fan_in,fan_out):
        sd = np.sqrt(2/(fan_in + fan_out))
        init_weight_tensor = np.random.randn(fan_in,fan_out) * sd
        return init_weight_tensor
class He:
    def initialize(self,weights_shape,fan_in,fan_out):
        sd = np.sqrt(2/fan_in)
        init_weight_tensor = np.random.randn(fan_in,fan_out) * sd
        return init_weight_tensor


