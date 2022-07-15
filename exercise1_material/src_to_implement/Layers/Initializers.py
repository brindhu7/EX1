import numpy as np
class Constant:
    def __init__(self,x=0.1):
        self.x = x

    def initialize(self,weights_shape,fan_in,fan_out):
        init_weight_tensor = np.ones(weights_shape) * self.x
        return init_weight_tensor


class UniformRandom:

    def initialize(self,weights_shape,fan_in,fan_out):
        init_weight_tensor = np.random.uniform(0,1,size=(weights_shape))
        return init_weight_tensor

class Xavier:

    def initialize(self,weights_shape,fan_in,fan_out):
        sd = np.sqrt(2/(fan_in + fan_out))
        init_weight_tensor = np.random.randn(fan_in,fan_out) * sd
        return init_weight_tensor
class He:
    def initialize(self,weights_shape,fan_in,fan_out):
        sd = np.sqrt(2/fan_in)
        init_weight_tensor =  np.random.normal(size=weights_shape, scale = sd)



        '''
        if len(weights_shape) == 4:
            init_weight_tensor = np.random.randn(weights_shape[0], weights_shape[1], weights_shape[2], weights_shape[3])
        elif len(weights_shape) == 3:
            init_weight_tensor = np.random.randn(weights_shape[0], weights_shape[1], weights_shape[2])
        elif len(weights_shape) == 2:
            init_weight_tensor = np.random.randn(weights_shape[0], weights_shape[1])
        else:
            init_weight_tensor = np.random.randn(weights_shape[0], 1)
        #init_weight_tensor =  np.random.normal(size=weights_shape, scale = sd)
        return init_weight_tensor * sd
        '''
        return init_weight_tensor
#

