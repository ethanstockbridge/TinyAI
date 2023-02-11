import numpy as np

class ReLU():
    """
    Encompasses ReLU functionality 
    """
    def calc(self, Z):
        """Calculate the ReLU on the input, which returns 0
        if Z is negative, or Z if Z is positive

        Args:
            Z (numpy array): Input array

        Returns:
            numpy array: Output array
        """
        return np.maximum(0, Z)

    def deriv(self, Z):
        """Undo the ReLU function, do the derivative, so return
        return where the numbers are positive.

        Args:
            Z (numpy array): Input array

        Returns:
            numpy array: Output array
        """
        return Z > 0