# "TinyAI" - Gradient Descent 

## Introduction

TinyAI is an AI developed from the ground up for educational purposes. Rather than relying on pre-existing AI code from online documentation, the aim of this project is to build the basic functionality of an AI neural network from scratch, in order to develop a deeper understanding of its workings at a low-level. It is not designed to compete with the speed and functionality of other neural network plugins like [Tensorflow](www.tensorflow.org) or [PyTorch](www.pytorch.org).

## Background 

Gradient Descent is an iterative machine learning algorithm that determines the optimal values for internal weight and bias parameters. This approach is widely used in artificial intelligence, particularly when parameters cannot be calculated analytically using linear algebra. Its effectiveness lies in its efficiency and straightforward use of specified inputs and outputs. TinyAI leverages this technique to identify the optimal weights and biases that represent the output, based on the given input and neural network specifications.

## Usage

You can test TinyAI's neural network by opening the main.py file. Here, you will find an example test utilizing the dataset brough by the [mnist](https://www.tensorflow.org/datasets/catalog/mnist) dataset. In this example, it utilizes 3 layers, however this can be easily changed to the amount of layers and neurons you desire. For custom usage, initialize a NeuralNetwork object and add Layers to it, specifying the neuron count, activation function, and alpha value. The goal was to make it as easy as possible to utilize this with only a few inputs. 

## Analysis

The analysis section will utilize the mnist dataset presented earlier to demonstrate the capabilities of TinyAI.

### Accuracy

Testing with a input layer of 784 (28x28 image from the mnist dataset), dense layer of 10 neurons, output dense layer of 10 neurons, learning rate of 0.1:

<img src="./res/accuracy-784-10-10-0.1.png" width="360"/>

After only 100 epochs, the accuracy can reach greater than 72 percent confidence at identifying hand-drawn numbers from the mnist dataset.  
After 1000 epochs, the accuracy increases to greater than 88 percent.  

### Mean squared error

The following tests were conducted with the same specifications as the aforementioned trials, and the corresponding Mean Squared Error (MSE) values are displayed below. As is typically observed with loss graphs, the MSE begins at a high value and steadily decreases over time, with the majority of the reduction occurring between epoch marks 0 to 200. This trend is consistent with the accuracy graph displayed above. It is reasonable to expect this relationship, given that MSE is a measure of the error when comparing the calculated output to the desired output.

<img src="./res/MSE-784-10-10-0.1.png" width="360"/>

### Learning rates

It is important to note that by adjusting the alpha value, we are essentially placing a constraint on the speed of learning within the neural network. While a very low learning rate may enable faster learning, it can also be less efficient. Conversely, assuming that a higher value will lead to better results is not necessarily true. In fact, using a higher value may initially lead to a significant improvement in accuracy, but may ultimately result in a sudden drop-off as the values go completely out of range. During gradient descent, this represents a rapid change in the approximation, which destroys the current understanding of the system. Please refer to the visual representation of different learning rates below:

<img src="./res/accuracy-784-10-10.png" width="360"/> 

It is worth noting that a higher learning rate can indeed be beneficial, up to a certain point. Using a very low learning rate can lead to inefficiencies, whereas using an excessively high value can result in the accuracy of the neural network dropping off, as demonstrated earlier. Therefore, finding an optimal learning rate that enables the neural network to achieve a fast increase in accuracy without encountering sudden drop-offs is crucial.

### Layer sizes

The size of a layer is another crucial factor that can impact the accuracy of a neural network over multiple epochs. However, it is not necessarily true that having more neurons in a dense layer will always result in better performance. As an example, consider a system with 784 input neurons, x hidden neurons, and 10 output neurons. By varying the number of hidden layers, we can evaluate the effect on the system's accuracy over 1000 epochs, as demonstrated below:

<img src="./res/accuracy-784-x-10.png" width="360"/> 

|Hidden layer size|Time required|
|----|----|
|5|164.064s|
|10|180.835s|
|50|309.565s|
|100|388.003s|

As shown above, the accuracy slope of the neural network increases most quickly with 100 neurons, resulting in an accuracy of over 92% after 1000 epochs. However, this trial also takes the longest and requires the most mathematical calculations, which incurs a trade-off between accuracy and resource usage. Ultimately, it is up to the user to decide which trade-off to prioritize.  
Another important factor to consider is that the hidden layers may not have sufficient neurons to accurately represent the input. For instance, if the output requires 10 distinct options (0-9), but the hidden layer only has two neurons, it is impossible to discern 10 distinct states from a linear combination of two.

### Layer count

Similarly to layer size, layer count is also a crucial factor in determining the accuracy of a neural network. However, with an increased layer count, the learning rate should decrease. The learning rate determines the sensitivity of how quickly the neural network learns, and a higher layer count can lead to an increased learning rate. If both values are increased, it can result in the slope being too steep, causing overshooting and a collapse of accuracy. This phenomenon has been observed when using a layer count of 3x10 hidden layers and a learning rate of 0.05, which was found to be well below the maximum learning rate before collapse, as noted in the above section. In addition, an increase in layer count will also increase the time and computational resources required to produce a resulting model, as demonstrated in the findings above on layer size.

## Conclusion

In summary, TinyAI is a valuable resource for educational purposes, providing a practical way to gain experience in the field of neural networks. This project represents just a fraction of what can be achieved with AI, and by exploring topics such as learning rate, layer sizes, neuron counts, and activation functions, one can gain a deeper understanding of how AI functions at a fundamental level. Overall, TinyAI is an excellent tool for anyone interested in learning more about artificial intelligence.