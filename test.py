import os.path
import random

import numpy as np
from matplotlib import pyplot as plt

import mnist
from network import NeuralNetwork

# Download dataset
if(not os.path.exists('mnist.pkl')): mnist.get()

# Load dataset
training_data, training_labels, testing_data, testing_labels = mnist.load()

# Create NN
nn = NeuralNetwork(784, 100, 10)

# Train
nn.train(training_data, training_labels, 1200, 50, 0.02)

# Print acc and plot perf
acc = nn.accuracy(training_data, training_labels)
print('Train Accuracy:', acc)
acc = nn.accuracy(testing_data, testing_labels)
print('Test Accuracy:', acc)
nn.plot_performance()

# Pick a random example from testing data
index = random.randint(0, 9999)

# Show the test data and the label
plt.imshow(training_data[index].reshape(28, 28))
plt.show()

# Show prediction
nn.feed(training_data[index])
model_output = nn.get_output()
print('Predicted: ', np.argmax(model_output))
