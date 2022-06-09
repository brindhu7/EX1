import numpy as np
class Sgd:
    def __init__(self,learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self,weight_tensor,gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights
class SgdWithMomentum:
    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self,weight_tensor,gradient_tensor):
        self.v = ( self.momentum_rate * self.v ) - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.v
        return updated_weights

class Adam:
    def __init__(self,learning_rate,mu,rho):
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
        self.r = self.rho * self.r + (1- self.rho ) * np.dot (gradient_tensor,gradient_tensor)
        #removing the bias in the weighted average
        predicted_v = self.v / (1 - np.power(self.mu, self.k))
        predicted_r = self.r / (1. - np.power(self.rho, self.k))
        weight_tensor = weight_tensor - self.learning_rate * (predicted_v / (np.sqrt(predicted_r) + self.epsilon))
        #updating the weights for every iteration
        self.k += 1
        return weight_tensor




