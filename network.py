import numpy as np
import tqdm
from matplotlib import pyplot as plt

from layer import Layer
from functions import relu_prime, cross_entropy_prime, softmax_prime


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initializing hidden and the output layer
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)

    # feed forward
    def feed(self, input):
        self.input_layer = input
        self.hidden_layer.feed_forward_relu(self.input_layer)
        self.output_layer.feed_forward_softmax(self.hidden_layer.activations)

    # backpropagation 
    def _backprob(self, label):
        self.output_bias_delta = cross_entropy_prime(self.output_layer.activations, label) * softmax_prime(self.output_layer.activations)
        self.hidden_bias_delta = (self.output_bias_delta @ self.output_layer.weights) * relu_prime(self.hidden_layer.weighted_sum)
        self.output_weight_delta = np.multiply.outer(self.output_bias_delta, self.hidden_layer.activations)
        self.hidden_weight_delta = np.multiply.outer(self.hidden_bias_delta, self.input_layer)

    def get_output(self):
        return self.output_layer.activations

    def _out_size(self):
        return self.output_layer.size

    def _train_batch(self, batch, batch_labels):
        self.tot_output_bias_delta = 0
        self.tot_hidden_bias_delta = 0
        self.tot_output_weight_delta = 0
        self.tot_hidden_weight_delta = 0

        for i in range(batch_labels.shape[0]):
            self.feed(batch[i])
            self._backprob(batch_labels[i])
            self.tot_output_bias_delta += self.output_bias_delta
            self.tot_hidden_bias_delta += self.hidden_bias_delta
            self.tot_output_weight_delta += self.output_weight_delta
            self.tot_hidden_weight_delta += self.hidden_weight_delta

    def train(self, training_data, labels, epochs, batch_size, learning_rate):
        self._perf_log = []

        pbar = tqdm.trange(epochs, ncols=80)
        for epoch in pbar:
            start_index = (epoch * batch_size) % training_data.shape[0]
            end_index = start_index + batch_size
            indices = np.arange(start_index, end_index) % training_data.shape[0]
        
            batch = training_data[indices]
            batch_labels = labels[indices]

            self._train_batch(batch, batch_labels)
            self.hidden_layer.update(self.tot_hidden_bias_delta,self.tot_hidden_weight_delta, learning_rate)
            self.output_layer.update(self.tot_output_bias_delta, self.tot_output_weight_delta, learning_rate)
        
            accuracy = self.accuracy(training_data, labels)
            self._perf_log.append(self.accuracy(training_data, labels))
            pbar.set_postfix(acc=accuracy)

    def plot_performance(self):
        plt.figure('Performance Graph')

        plt.plot(self._perf_log)

        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')

        plt.show()

    def accuracy(self, testing_data, testing_targets):
        # Evalute over all the testing data and get outputs
        self.feed(testing_data)

        # Create a onehot array from outputs
        output_onehot = self.get_output_onehot()

        # Calculate how many examples it classified correctly  
        no_correct = (testing_targets == output_onehot).all(axis=1).sum()
        
        # Calculate accuracy
        accuracy = (no_correct/testing_data.shape[0]) * 100

        # Return accuracy
        return round(accuracy, 2) 

    def get_output_onehot(self):
        output_targets = self.get_output()

        if(output_targets.ndim == 1):
            # If output is a vector/1D array, axis=1 will not work
            index = np.argmax(output_targets)
            output_onehot = np.zeros((self._out_size))
            output_onehot[index] = 1
        else:
            # Create a onehot array from outputs
            output_onehot = np.zeros(output_targets.shape)
            output_onehot[np.arange(output_targets.shape[0]), np.argmax(output_targets, axis=1)] = 1

        # Return one hot array
        return output_onehot

