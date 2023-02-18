import numpy as np
import json

def one_hot(Z):
    """Returns numpy array with the highest index as 1 and the
    rest as 0

    Args:
        Z (numpy array): Numpy array of floats 

    Returns:
        numpy array: Postprocessed array
    """
    one_hot_Z = np.zeros((Z.size, Z.max() + 1))
    one_hot_Z[np.arange(Z.size), Z] = 1
    one_hot_Z = one_hot_Z.T
    return one_hot_Z

def arrange_data(data, split):
    """Get the data from a location

    Args:
        data (numpy array): Data
        split (float): Percent train:test
        0.7 means 70% will be used for train

    Returns:
        train, test: Training and testing data
    """
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    train_m = int(m*split)

    data_train = data[0:train_m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    _, m_train = X_train.shape

    data_test = data[train_m:].T
    Y_test = data_test[0]
    X_test = data_test[1:n]
    _, m_dev = X_test.shape
    return X_train.T, Y_train, X_test.T, Y_test