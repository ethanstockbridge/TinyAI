"""

Prodides a base class to each of the layer types.

"""

class Layer:
    """Base class Layer
    """
    def __init__(self, neurons):
        """Initialize parent

        Args:
            neurons (int): Number of neurons
        """
        self.neurons=neurons
        self.id = None
    def link_layers(self, prev, nxt):
        """Link the previous and next layers

        Args:
            prev (Layer): Previous layer
            nxt (Layer): Next layer
        """
        self.prev = prev
        self.next = nxt