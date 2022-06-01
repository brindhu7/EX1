import copy

class NeuralNetwork:
    def __init__(self,Optimizer):
        self.Optimizer = Optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor,label_tensor = self.data_layer.next()
        self.input_tensor = input_tensor
        self.label_tensor  = label_tensor
        #calling forward path for all the layers and return loss in the end
        for i in self.layers:
            self.input_tensor = i.forward(self.input_tensor)
        return self.loss_layer.forward(self.input_tensor, self.label_tensor)

    def backward(self):
        #pass the error _tensor from loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)
        #calling backward pass for all layers, starting from the end(softmax)
        for i in self.layers[::-1]:
            error_tensor = i.backward(error_tensor)

    def append_layer(self,layer):
        #creating layers in the list layer
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.Optimizer)
        self.layers.append(layer)

    def train(self,iterations):
        #calling forward and backward for each iterations
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self,input_tensor):
        #input tensor for every layer is the output of the previous tensor
        for i in self.layers:
            input_tensor= i.forward(input_tensor)
        return input_tensor








