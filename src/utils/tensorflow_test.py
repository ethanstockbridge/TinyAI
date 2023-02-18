import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import time

max_epochs = 1000
batch_size = 1000

# Define a callback to stop training once validation accuracy reaches 90%
class MyCallback(tf.keras.callbacks.Callback):
    def on_test_begin(self,logs=None):
        global start_time
        if start_time==None:
            start_time = time.time()
    def on_test_batch_end(self, batch, logs=None):
        global start_time
        if logs.get('accuracy') is not None and logs.get('accuracy') >= 0.90:
            self.model.stop_training = True
            print("\nReached 90% validation accuracy, stopping training!")


if __name__ == "__main__":
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation="softmax")
    ])

    # Compile the model with the specified loss function and optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    global start_time
    start_time = None

    # Train the model on the training data with the callback
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=max_epochs, validation_data=(x_test, y_test), 
                        callbacks=[MyCallback()])

    end_time = time.time()

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('Time taken:', end_time - start_time, 'seconds')
