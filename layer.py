import numpy as np
from functions import relu, softmax

class Layer:
    def __init__(self, input_size, size):
        # initializing weights and bias
        self.size = size
        epsilon = np.sqrt(6 / (size + input_size))
        self.weights = np.random.uniform(-epsilon, epsilon, (size, input_size))
        self.bias = np.random.uniform(-epsilon, epsilon, (size))

    def feed_forward_relu(self, inp):
        self.weighted_sum = (inp @ self.weights.T) + self.bias
        self.activations = relu(self.weighted_sum)

    def feed_forward_softmax(self, inp):
        self.weighted_sum = (inp @ self.weights.T) + self.bias
        self.activations = softmax(self.weighted_sum)

    def update(self, bias_delta, weight_delta, learning_rate):
        self.weights -= (learning_rate * weight_delta)
        self.bias -= (learning_rate * bias_delta)
