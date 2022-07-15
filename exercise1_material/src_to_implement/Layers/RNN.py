from exercise1_material.src_to_implement.Layers import TanH
from exercise1_material.src_to_implement.Layers import Sigmoid
from exercise1_material.src_to_implement.Layers.Base import BaseLayer
from exercise1_material.src_to_implement.Layers import FullyConnected
import numpy as np


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        super().__init__()
        self.trainable = True
        self.memorize = False
        self.hidden_state = None
        self.fc_obj_hidden = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.tanh_obj = TanH.TanH()
        self.fc_obj_y = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.sigmoid_obj = Sigmoid.Sigmoid()
        self.firstbatch = True
        self.gradient_weights = None
        self.back_first = True
        self._optimizer = None
        self.fc_hidden_a = None



    @property
    def _memorize(self):
        return self.memorize

    @_memorize.setter
    def _memorize(self, value):
        self.memorize = value

    def forward(self, input_tensor):
        # initialize hidden state
        self.b, ip = input_tensor.shape
        self.input_tensor = input_tensor
        if self.memorize is False or self.firstbatch:
            self.hidden_state = np.zeros([self.b, self.hidden_size])
            self.firstbatch = False

        # initializing output_tensor(Y_tensor)
        self.y_tensor = np.zeros([self.b, self.output_size   ])
        #initializing the variables to store the input tensor to be used in backward path
        self.fc_hidden_a = np.zeros([self.b, self.input_size + self.hidden_size ])
        self.tanh_a = np.zeros([self.b, self.hidden_size])
        self.fc_y_a = np.zeros([self.b, self.hidden_size])
        self.sigmoid_a = np.zeros([self.b, self.output_size])


        # iterate over every time stamp
        for t,x in enumerate(input_tensor):
            x_hat = np.concatenate((x, self.hidden_state[t - 1]))  ## dimension - (20,)
            # x_hat as input_tensor to fully connected layer. fully_connected layer input_tensor dimension -Batch_size,input_size
            # converting x_hat to 1(batch) * 20(input_dimension)
            inp_for_TanH = self.fc_obj_hidden.forward(np.expand_dims(x_hat, axis=0))
            #for correct gradient calculation
            self.fc_hidden_a[t] = self.fc_obj_hidden.input_tensor
            self.hidden_state[t] = self.tanh_obj.forward(inp_for_TanH)
            self.tanh_a[t] = self.tanh_obj.output

            # pass hidden(activation) state to next fullyconnected layer
            inp_for_sigmoid = self.fc_obj_y.forward(np.expand_dims(self.hidden_state[t], axis=0))
            self.fc_y_a[t] = self.fc_obj_y.input_tensor
            self.y_tensor[t] = self.sigmoid_obj.forward(inp_for_sigmoid)
            self.sigmoid_a[t] = self.sigmoid_obj.output

        return self.y_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.prev_error_tensor = np.zeros([self.b, self.input_size])
        self.hidden_state_back = np.zeros([self.b, self.hidden_size])
        #initially no gradient
        self.hidden_gradient= 0
        self.concatinated_gradients = 0

        #iterate from last error vector of the timestamp
        for i,error_t in enumerate(self.error_tensor[::-1]):
            t = self.b - i
            self.sigmoid_obj.output = self.sigmoid_a[t - 1]
            inp_for_fc_y = self.sigmoid_obj.backward(np.expand_dims(error_t, axis=0))
            self.fc_obj_y.input_tensor = np.expand_dims(self.fc_y_a[t - 1], axis=0)
            #first batch there is no previous hidden_state
            if i==0:
                inp_for_tanh = self.fc_obj_y.backward(inp_for_fc_y)
            else:
                inp_for_tanh = self.fc_obj_y.backward(inp_for_fc_y) + self.hidden_state_back[t]

            self.tanh_obj.output = self.tanh_a[t - 1]
            #getting gradient from FC_sigmoid
            self.hidden_gradient += self.fc_obj_y.gradient_weights
            self.tanh_obj.output = self.tanh_a[t-1]
            inp_for_fc = self.tanh_obj.backward(inp_for_tanh)
            self.fc_obj_hidden.input_tensor = np.expand_dims(self.fc_hidden_a[t - 1], axis=0)
            output = self.fc_obj_hidden.backward(inp_for_fc)
            #gradient from this 'self.fc_obj_hidden.gradient_weights' will be gradient with respect to xhat
            self.concatinated_gradients += self.fc_obj_hidden.gradient_weights

            self.prev_error_tensor[t-1] = output[:,:self.input_size]
            self.hidden_state_back[t-1] = output[:,self.input_size:]

        self.weights = self.fc_obj_hidden.weights
        self.gradient_weights = self.get_gradient_weights

        if self._optimizer:
            self.fc_obj_y.weights = self._optimizer.calculate_update(self.fc_obj_y.weights,self.hidden_gradient)
            self.fc_obj_hidden.weights = self._optimizer.calculate_update(self.fc_obj_hidden.weights,self.concatinated_gradients)

        return self.prev_error_tensor


    #getter property
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,value):
        self._optimizer = value

    @property
    def get_gradient_weights(self):
        return self.concatinated_gradients

    @property
    def weights(self):
        return self.fc_obj_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_obj_hidden.weights = value


    def initialize(self, weights_initializer, bias_initializer):
        self.fc_obj_hidden.initialize(weights_initializer,bias_initializer)
        self.fc_obj_y.initialize(weights_initializer,bias_initializer)





































