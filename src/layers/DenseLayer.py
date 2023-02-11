"""

Dense (intermediate) layer(s) of the neural network 


"""

from layers.Layer import Layer
import numpy as np
from utils.utils import softmax, one_hot

class DenseLayer(Layer):
    """Dense layer for the neural network. Connects all nodes from the previous
    layer to all nodes of this layer

    Args:
        Layer (Layer): Parent
    """

    def __init__(self, neurons, activation, alpha):
        """Initialize the dense layer

        Args:
            neurons (int): The number of neurons for this layer
            activation (class): Activation functions
            alpha (float): Alpha value
        """
        super().__init__(neurons)
        self.last_layer = False
        self.activation = activation
        self.id = None
        self.alpha = alpha
        pass

    def __str__(self):
        """Generate a string representation for the layer

        Returns:
            str: Representation
        """
        if self.last_layer:
            return "Output Dense layer. Neurons: "+str(self.neurons)
        return "Dense layer. Neurons: "+str(self.neurons)

    def init_params(self):
        """Initialize parameters by creating random inital numbers for weights and bias'
        """
        #random weights for each neuron to neuron linking previous to current layer
        self.weights = np.random.rand(self.neurons, self.prev.neurons) - 0.5
        #random biases for the output of this layer
        self.bias = np.random.rand(self.neurons, 1) - 0.5

    def forward_pass(self):
        """Perform a forward pass within the layer

        Returns:
            numpy array, numpy array, numpy array: Z values, Neuron data, weights
        """
        self.Z = self.weights.dot(self.prev.neuron_data) + self.bias
        self.neuron_data = self.activation.calc(self.Z)
        if self.last_layer:
            self.neuron_data=softmax(self.neuron_data)
        return self.Z, self.neuron_data, self.weights

    def backward_pass(self, X=None, Y=None):
        """Perform a backward pass within the layer

        Args:
            A_prev (numpy array, optional): Neuron results from the previous layer. Defaults to None.
            W_next (numpy array, optional): Weights of the next layer. Defaults to None.
            dZ_next (numpy array, optional): dZ of the next layer. Defaults to None.
            X (numpy array, optional): X set. Defaults to None.
            Y (numpy array, optional): Y set. Defaults to None.

        Returns:
            numpy array, numpy array, numpy array: Z values, weights, bias'
        """
        train_size = X.shape[1]
        if self.last_layer:
            one_hot_Y = one_hot(Y)
            self.dZ = self.neuron_data - one_hot_Y
            dW = 1 / train_size * self.dZ.dot(self.prev.neuron_data.T)
            dB = 1 / train_size * np.sum(self.dZ)
        else:
            self.dZ = self.next.weights.T.dot(self.next.dZ) * self.activation.deriv(self.Z)
            dW = 1 / train_size * self.dZ.dot(self.prev.neuron_data.T)
            dB = 1 / train_size * np.sum(self.dZ)
        self.update_params(dW, dB)
        return self.Z, self.weights, self.bias
    
    def update_params(self,dW,dB):
        """Update the parameters of weights and bias' after a back pass

        Args:
            dW (numpy array): dWeights
            dB (numpy array): dBias'
        """
        self.weights = self.weights - self.alpha * dW
        self.bias = self.bias - self.alpha * dB

    def link_layers(self, prev, nxt):
        """Link the layers together

        Args:
            prev (Layer): Previous layer
            next (Layer): Next layer
        """
        super().link_layers(prev, nxt)
        # print("here")
        self.init_params()