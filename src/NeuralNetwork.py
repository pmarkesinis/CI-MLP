# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron:
    def __init__(self, weight=[0.1, 0.1], bias=0.5, a=0.05, epoch=100):
        self.weight = weight
        self.bias = bias
        self.a = a
        self.epoch = epoch

    # STEP function
    def activationFunction(self, z):
        return 1 if z >= 0 else 0

    def makePrediction(self, x):
        z = np.dot(self.weight, x) + self.bias
        return self.activationFunction(z)

    def calcError(self, predicted, expected):
        return expected - predicted

    def updateValues(self, x, expected):
        errors = []
        for i in range(self.epoch):
            temp = 0
            for j in range(len(x)):
                predicted = self.makePrediction(x[j])
                loss = self.calcError(predicted, expected[j])
                temp += abs(loss)
                for k in range(len(self.weight)):
                    self.weight[k] += self.a * x[j][k] * loss
                self.bias += self.a * loss
            errors.append(temp)
        return errors


class MLP:

    # Initialization function
    # Input size is the number of input neurons
    # Hidden layers is a 1d array of the numbers of neurons per hidden layer
    # Output size is the number of output neurons

    def __init__(self, hidden_layers, input_size=10, output_size=7, a=0.005, epoch=35, batch_size=16):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.a = a
        self.epoch = epoch
        self.batch_size = batch_size

        self.structure = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases
        # self.biases = [np.random.randn(x, 1) for x in self.structure[1:]]
        self.biases = [np.zeros((x, 1), dtype=float) for x in self.structure[1:]]
        for i in range(len(self.biases)):
            self.biases[i] = np.transpose(self.biases[i])

        self.weights = [np.random.randn(x, y) for (x, y)
                        in zip(self.structure[1:], self.structure[:-1])]

    # Sigmoid Function for activation
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_function_derivative(self, z):
        return self.activation_function(z) * (1 - self.activation_function(z))

    def cost_derivative(self, prediction, output_data):
        return (prediction - output_data)

    # assume y_true and y_pred are arrays of shape (batch_size, output_size)
    # def mse_loss(self, expected, predicted):
    #     mse = np.mean((expected - predicted) ** 2)
    #     return mse

    # Categorical Cross Entropy for a loss function
    def categorical_crossentropy_loss(self, predicted, expected):

        # expected = np.concatenate([expected, np.zeros((expected.shape[0], 2))], axis=1)
        loss = -np.sum(expected * np.log(predicted))
        return loss

    def forward_propagation(self, input_data):
        activation = input_data
        for i in range(len(self.weights)):
            if (i == len(self.weights) - 1):
                z = np.dot(activation, self.weights[i].T)
            else:
                z = np.dot(activation, self.weights[i].T)
            z = z + self.biases[i]
            activation = self.activation_function(z)
        return activation
    
    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            output = self.forward_propagation(data[i])
            predictions.append(output)
        return predictions

    def setEpoch(self, new):
        self.epoch = new

    def backward_propagation(self, input_data, output_data, learning_rate):
        activations = [input_data]
        zs = []
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                z = np.dot(activations[i], self.weights[i].T) + self.biases[i]
            else:
                z = np.dot(activations[i], self.weights[i].T) + self.biases[i]
            zs.append(z)
            output = self.activation_function(z)
            activations.append(output)
        prediction = activations[-1]

        delta = self.cost_derivative(prediction, output_data) * \
                self.activation_function_derivative(zs[-1])
        gradients_w = [np.dot(activations[-2].T, delta)]
        gradients_b = [np.sum(delta, axis=0)]
        for l in range(2, len(self.structure)):
            z = zs[-l]
            delta = np.dot(delta, self.weights[-l + 1]) * \
                    self.activation_function_derivative(z)
            gradients_w.append(np.dot(activations[-l - 1].T, delta))
            gradients_b.append(np.sum(delta, axis=0))
        gradients_w.reverse()
        gradients_b.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                self.weights[i] -= learning_rate * gradients_w[i].T
            else:
                self.weights[i] -= learning_rate * gradients_w[i].T
            self.biases[i] -= learning_rate * gradients_b[i]

    def train(self, training_set, target_set, validation_set, validation_targets):
        batches_num = int(np.ceil(len(training_set) / self.batch_size))
        training_losses = []
        validation_losses = []

        for i in range(self.epoch):

            # Shuffle the data
            shuffled_indices = np.random.permutation(len(training_set))
            training_set = training_set[shuffled_indices]
            target_set = target_set[shuffled_indices]

            for batch_index in  range(batches_num):
                start = batch_index * self.batch_size
                end = (batch_index + 1) * self.batch_size
                mini_batch_set = training_set[start:end]
                mini_batch_targets = target_set[start:end]
                train_predictions = self.forward_propagation(mini_batch_set)
                train_loss = self.categorical_crossentropy_loss(train_predictions, mini_batch_targets)
                training_losses.append(train_loss)

                validation_predictions = self.forward_propagation(validation_set)
                validation_loss = self.categorical_crossentropy_loss(validation_predictions, validation_targets)
                validation_losses.append(validation_loss)

                self.backward_propagation(mini_batch_set, mini_batch_targets, self.a)
            #     np.save(f"62_model_weights_epoch{i+1}.npy", self.weights[i % len(self.weights)])
            #
            # if (i+1) % 2 == 0:
            #     np.save(f"MLP_weights_epoch_{i + 1}.npy", self.weights[i % len(self.weights)])
            #     np.save(f"MLP_biases_epoch_{i + 1}.npy", self.biases[i % len(self.weights)])

            # print(f"Epoch {i} completed.")

        # plt.plot(training_losses, label="Training Losses")
        # plt.plot(validation_losses, label="Validation Losses")
        # plt.legend()
        # plt.title("Training and Validation Losses")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.show()

class ANN:
    def __init__(self):
        pass
