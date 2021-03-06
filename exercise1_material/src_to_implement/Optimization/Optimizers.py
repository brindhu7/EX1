import numpy as np
class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self,learning_rate):
        super().__init__()
        self.learning_rate = float(learning_rate)

    def calculate_update(self,weight_tensor,gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return updated_weights

class SgdWithMomentum(Optimizer):
    def __init__(self,learning_rate,momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self,weight_tensor,gradient_tensor):
        self.v = ( self.momentum_rate * self.v ) - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.v
        if self.regularizer:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return updated_weights

class Adam(Optimizer):
    def __init__(self,learning_rate,mu,rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon =  np.finfo(float).eps
        self.v = 0
        self.k = 1
        self.r = 0

    def calculate_update(self,weight_tensor,gradient_tensor):
        #weighted average for momentum by taking the past gradients
        self.v = self.mu * self.v + (1- self.mu) * gradient_tensor
        #self.r = self.rho * self.r + (1- self.rho ) * np.dot (gradient_tensor,gradient_tensor)

        self.r = self.rho * self.r + ((1 - self.rho) * gradient_tensor)*gradient_tensor
        #removing the bias in the weighted average
        predicted_v = self.v / (1 - np.power(self.mu, self.k))
        predicted_r = self.r / (1. - np.power(self.rho, self.k))
        updated_weights = weight_tensor - self.learning_rate * (predicted_v / (np.sqrt(predicted_r) + self.epsilon))
        #updating the weights for every iteration
        self.k += 1
        if self.regularizer:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return updated_weights





