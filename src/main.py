import numpy as np
import pandas as pd
from tqdm import tqdm
from NeuralNetwork import NeuralNetwork
from activation.ReLU import ReLU
from activation.SoftMax import SoftMax
from utils.utils import arrange_data
from layers.DenseLayer import DenseLayer
from layers.InputLayer import InputLayer
import time
import os

# Configuration
BATCH_SIZE = 5000
EPOCHS = 100
ALPHA = 0.1
TENSORFLOW_DATASET = True

if __name__ == "__main__":
    """Utilize the Mnist dataset to run on the TinyAI custom neural network
    """

    print("Preparing data")
    if(TENSORFLOW_DATASET):
        # extract data from tensorflow's keras mnist
        from tensorflow.keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    else:
        # or from a manual dataset:
        if not os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","res","train.csv")):
            print("Could not find file train.csv, ensure you have it downloaded in the resources folder, \"res\"")
            exit(1)
        else:
            data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","res","train.csv"))
            X_train, Y_train, X_test, Y_test = arrange_data(data, 0.7)
            print(X_train.shape)
            print(Y_train.shape)

    # noralize to 0-1
    X_train = np.array([np.array(x.flatten()/225) for x in X_train])
    X_test = np.array([np.array(x.flatten()/225) for x in X_test])
    # split the data into batches
    X_train_batch = [X_train[i:i+BATCH_SIZE].T for i in range(0, len(X_train), BATCH_SIZE)]
    X_test_batch = [X_test[i:i+BATCH_SIZE].T for i in range(0, len(X_test), BATCH_SIZE)]
    Y_train_batch = [Y_train[i:i+BATCH_SIZE] for i in range(0, len(Y_train), BATCH_SIZE)]
    Y_test_batch = [Y_test[i:i+BATCH_SIZE] for i in range(0, len(Y_test), BATCH_SIZE)]

    nn = NeuralNetwork()
    #add your input layer
    nn.add(InputLayer(784, X_train, Y_train))
    #add your intermediate layers
    nn.add(DenseLayer(10, ReLU(), ALPHA))
    #add your output layer
    nn.add(DenseLayer(10, SoftMax(), ALPHA))

    print("Created neural network:")
    print(nn)
    print("Training network now:")

    start_time = time.time()
    
    # Train fast, somewhat inaccurate relative to entire dataset.
    tqdm_itr = tqdm(range(0,EPOCHS))
    for x in tqdm_itr:
        for X_set, Y_set in zip(X_train_batch, Y_train_batch):
            nn.iterate(X_set, Y_set)
            accuracy = nn.accuracy()
            tqdm_itr.set_description(f"Accuracy: {accuracy:0.3}%  Progress")

    # Enable the below for a second, more accuracte training
    # Train against whole dataset for accuracy
    # EPOCHS = 900
    # tqdm_itr = tqdm(range(0,900))
    # for x in tqdm_itr:
    #     nn.iterate(X_train.T, Y_train)
    #     tqdm_itr.set_description(f"Accuracy: {nn.accuracy():0.3}%  Progress")
    
    end_time = time.time()

    print(f"Training took: {(end_time-start_time):0.3} seconds")
    print(f"Accuracy on x_train: {nn.accuracy():0.3}%")
    print(f"Accuracy on x_test: {nn.accuracy(X_test.T, Y_test):0.3}%")

    print("Test Predictions | Test Real")
    print(nn.predict(X_test[0]), Y_test[0])
    print(nn.predict(X_test[1]), Y_test[1])
    print(nn.predict(X_test[2]), Y_test[2])
    print(nn.predict(X_test[3]), Y_test[3])
    print(nn.predict(X_test[4]), Y_test[4])
