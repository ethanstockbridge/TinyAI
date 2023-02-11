import numpy as np
import pandas as pd
from tqdm import tqdm
from NeuralNetwork import NeuralNetwork
from activation.ReLU import ReLU
from utils.utils import arrange_data
from layers.DenseLayer import DenseLayer
from layers.InputLayer import InputLayer
from tensorflow.keras.datasets import mnist

if __name__ == "__main__":
    """Utilize the Mnist dataset to run on the TinyAI custom neural network
    """

    # extract data from tensorflow's keras mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.array([np.array(x.flatten()/225) for x in X_train]).T
    X_test = np.array([np.array(x.flatten()/225) for x in X_test]).T
    # or from a manual dataset:
    # data = pd.read_csv("./train.csv")
    # X_train, Y_train, X_test, Y_test = arrange_data(data, 0.7)
    alpha = 0.1

    nn = NeuralNetwork()
    #add your input layer
    nn.add(InputLayer(784, X_train, Y_train))
    #add your intermediate layers
    nn.add(DenseLayer(10, ReLU(), alpha))
    #add your output layer
    nn.add(DenseLayer(10, ReLU(), alpha))

    print("Created neural network:")
    print(nn)

    print("Training network now:")
    tqdm_itr = tqdm(range(0,1000))
    for x in tqdm_itr:
        nn.iterate()
        tqdm_itr.set_description(f"Accuracy: {nn.accuracy():0.3}%  Progress")

    print(f"Accuracy on x_test: {nn.accuracy():0.5}%")
    print(f"Accuracy on x_train: {nn.accuracy(X_train):0.3}%")

    print("Test Predictions | Test Real")
    print(nn.predict(X_test.T[0]), Y_test[0])
    print(nn.predict(X_test.T[1]), Y_test[1])
    print(nn.predict(X_test.T[2]), Y_test[2])
    print(nn.predict(X_test.T[3]), Y_test[3])
    print(nn.predict(X_test.T[4]), Y_test[4])
    print(nn.predict(X_test.T[5]), Y_test[5])
