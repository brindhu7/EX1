from exercise1_material.src_to_implement.Optimization.Optimizers import Optimizer
import numpy as np

class L1_Regularizer(Optimizer):
    def __init__(self, alpha):
        super().__init__()
        self.add_regularizer(alpha)

    def calculate_gradient(self, weights):
        return self.regularizer*np.sign(weights)

    def norm(self, weights):
        return np.sum(np.abs(weights))*self.regularizer

class L2_Regularizer(Optimizer):
    def __init__(self, alpha):
        super().__init__()
        self.add_regularizer(alpha)

    def calculate_gradient(self, weights):
        return self.regularizer*weights

    def norm(self, weights):
        return np.sum(np.abs(weights)**2)*self.regularizer