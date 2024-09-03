import pickle
import numpy as np
from util import *

class MultilayerPerceptron:
    def __init__(self, learning_rate=0.1, num_epochs=100, reg_lambda=0):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.reg_lambda = reg_lambda  # Regularization strength
        self.weights = None

    def initialize_weights(self, input_size, hidden_units, output_size):
        # He Initialization
        self.weights = [
            np.random.randn(input_size, hidden_units) * np.sqrt(2. / input_size),
            np.random.randn(hidden_units, output_size) * np.sqrt(2. / hidden_units)
        ]


    def fit(self, X, T, max_iters=1000):
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        input_size = X.shape[1]
        output_size = T.shape[1]
        hidden_units = 10  # Number of hidden units
        if (self.num_epochs > max_iters):
            self.num_epochs = max_iters

        # Initialize weights
        self.initialize_weights(input_size, hidden_units, output_size)

        # Training loop
        for epoch in range(self.num_epochs):
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            # Stochastic gradient descent
            for index in indices:
                x = X[index]
                t = T[index]

                # Forward pass
                hidden_layer_input = np.dot(x, self.weights[0])
                hidden_layer_output = np.tanh(hidden_layer_input)
                logits = np.dot(hidden_layer_output, self.weights[1])
                logits = logits.reshape(-1, logits.shape[-1])
                predictions = softmax(logits)

                # Backward pass (gradient computation)
                error = predictions - t
                hidden_error = np.dot(error, self.weights[1].T) * (1 - np.power(hidden_layer_output, 2))
                gradient_output = np.outer(hidden_layer_output, error)
                gradient_hidden = np.outer(x, hidden_error)

                # Update weights

                self.weights[0] -= self.learning_rate * gradient_hidden
                self.weights[1] -= self.learning_rate * gradient_output

    def predict(self, X):
        # Perform forward pass to get predictions
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        hidden_layer_output = np.tanh(np.dot(X, self.weights[0]))
        logits = np.dot(hidden_layer_output, self.weights[1])
        predictions = softmax(logits)
        return predictions

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    @staticmethod
    def load(path):
        model = MultilayerPerceptron()
        with open(path, 'rb') as f:
            model.weights = pickle.load(f)
        return model
