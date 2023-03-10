import numpy as np
from layers.Layer import Layer
from layers.InputLayer import InputLayer
from layers.DenseLayer import DenseLayer
from activation.ReLU import ReLU
from utils.utils import one_hot

class NeuralNetwork:
    """Neural network that holds the layers and performs passes and
    calculations
    """

    def __init__(self):
        """Initialize the neural network with defaults
        """
        self.layers = []
        self.link_status = False
        self.highest_accuracy = 0

    def __str__(self):
        """Generate a string representation of the neural network

        Returns:
            str: Representation
        """
        ss = ""
        for layer in self.layers:
            ss+=str(layer)+"\n"
        return ss[:-1]

    def add(self, layer):
        """Add a new layer to the network

        Args:
            layer (Layer): New layer

        Raises:
            Exception: Error if not layer
        """
        if Layer not in layer.__class__.__bases__:
            raise Exception("Error!")
        else:
            if len(self.layers)>0:
                self.layers[-1].last_layer = False
            self.layers.append(layer)
            self.layers[-1].last_layer = True

    def link_layers(self):
        """TODO: move functionality into add(self,)
        Link the layers
        """
        for i in range(0,len(self.layers)):
            self.layers[i].id = i
        for i in range(0,len(self.layers)):
            if i==0:
                self.layers[i].link_layers(None, self.layers[i+1])
            elif i==len(self.layers)-1:
                self.layers[i].link_layers(self.layers[i-1], None)
            else:
                self.layers[i].link_layers(self.layers[i-1], self.layers[i+1])
        self.link_status=True

    def forward_pass(self, X_set=None, Y_set=None):
        """Perform a forward pass from layer 0 to layer n

        Returns:
            numpy array, numpy array, numpy array: Last: Z, neuron data, weights
        """
        if not self.link_status:
            self.link_layers()
        if(X_set is not None):
            self.layers[0].neuron_data = X_set
        if(Y_set is not None):
            self.layers[0].y_set = Y_set
        for layer in self.layers:
            if layer and layer.__class__ is not InputLayer:
                layer.forward_pass()

    def backward_pass(self):
        """Perform a backwards pass from layer n to 0 and update the weights/bias'
        """
        for i in range(len(self.layers)-1,-1,-1):
            layer = self.layers[i]
            if layer and layer.__class__ is not InputLayer:
                layer.backward_pass(X=self.layers[0].neuron_data, Y=self.layers[0].y_set)

    def extract_weights_bias(self):
        """Extracts the values to be saved for future usage

        Returns:
            list[numpy array]: List of the weights and bias of all layers
        """
        vals = []
        for layer in self.layers[1:]:
            thislayer = []
            thislayer.append(layer.weights)
            thislayer.append(layer.bias)
            vals.append(thislayer)
        return vals

    def iterate(self, X_set=None, Y_set=None):
        """Iterate through the network which involves performing a forward pass
        and then a backward pass, then updating the weights and bias'

        Args:
            X_set (numpy array, optional): Optional dataset to run the iteration on
            Y_set (numpy array, optional): Optional dataset to compare the output to
        """
        #forward pass, backward pass, save values
        if(X_set is not None and Y_set is not None):
            self.layers[0].y_set = Y_set
            self.layers[0].neuron_data = X_set
        self.forward_pass()
        self.backward_pass()
        if(self.accuracy()>self.highest_accuracy):
            self.highest_accuracy = self.accuracy()
            self.highest_checkpoint = self.extract_weights_bias()

    def accuracy(self, X_set=None, Y_set=None):
        """Gets the accuracy of the network after training, or use the dataset
        to find the network's accuracy

        Args:
            dataset (numpy array, optional): Optional dataset to find the accuracy on. Defaults to None.

        Returns:
            float: Accuracy
        """
        if(X_set is not None and Y_set is not None):
            self.forward_pass(X_set = X_set, Y_set = Y_set)
        prediction = np.argmax(self.layers[-1].neuron_data, 0)
        accuracy = np.sum(prediction == self.layers[0].y_set) / self.layers[0].y_set.size
        accuracy*=100
        return accuracy

    def predict(self, X_set=None):
        """Predict the value of the dataset using the trained neural network

        Args:
            dataset (numpy array): Input dataset to parse

        Returns:
            Int: Prediction index
        """
        if X_set is not None:
            self.forward_pass(X_set = np.array([X_set]).T)
        prediction = np.argmax(self.layers[-1].neuron_data, 0)[0]
        return prediction

    def MSE(self):
        """Calculate the MSE of the neural network. Kinda slow but
        it works

        Returns:
            float: MSE
        """
        actual = self.layers[-1].neuron_data.T
        desired = self.layers[0].y_set
        desired = one_hot(desired).T
        sets = 0
        for i in range(0,len(actual)):
            this_actual = actual[i]
            this_desired = desired[i]
            MSE = 0
            for j in range(0,len(this_actual)):
                MSE+=pow((this_actual[j]-this_desired[j]),2)
            MSE/=len(this_actual)
            sets += MSE
        MSE = sets/len(actual)
        return MSE