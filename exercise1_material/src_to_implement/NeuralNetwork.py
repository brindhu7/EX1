
class NeuralNetwork:
    def __init__(self,Optimizer):
        self.Optimizer = Optimizer
        self.Loss = []
        self.Layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor,label_tensor = self.data_layer.next()




