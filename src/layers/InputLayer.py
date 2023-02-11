"""

Categorized as an input layer to the neural network

"""

from layers.Layer import Layer

class InputLayer(Layer):
    """Input layer for the neural network. Creates new neurons for all of
    the input data points in x_set

    Args:
        Layer (class): Parent
    """
    
    def __init__(self, neurons, x_set, y_set):
        """Initialize the dense layer

        Args:
            neurons (int): The number of neurons for this layer
            x_set (numpy array): Input data
            y_set (numpy array): Input data results
        """
        super().__init__(neurons)
        self.neuron_data = x_set
        self.y_set = y_set
        self.id = None

    def __str__(self):
        """Generate a string representation for the layer

        Returns:
            str: Representation
        """
        return "Input layer. Neurons: "+str(self.neurons)