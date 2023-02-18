import numpy as np

class SoftMax():
    """
    Encompasses SoftMax functionality 
    """
    def calc(self, Z):
        """Creates a softmax representation of the numpy array

        Args:
            Z (numpy array): Preproccessed numpy array, usually the
            output stage of neural network

        Returns:
            numpy array: Postprocessed array
        """
        A = np.exp(Z) / sum(np.exp(Z))
        return A
