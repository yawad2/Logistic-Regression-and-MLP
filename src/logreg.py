import numpy as np
import os
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=100, reg_lambda=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda  # Regularization strength
        self.weights = None
        self.bias = None
        self.accuracy_history = []

    def fit(self, X, T_one_hot_encoded):
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        input_size = X.shape[1]
        output_size = T_one_hot_encoded.shape[1]
        
        # Initialize weights and bias
        self.weights = np.random.normal(loc=0, scale=1, size=(input_size, output_size))
        self.bias = np.zeros(output_size)
        
        # Training loop
        for epoch in range(self.epochs):
            correct_predictions = 0
            
            # Stochastic gradient descent
            for x, t in zip(X, T_one_hot_encoded):
                # Forward pass
                logits = np.dot(x, self.weights) + self.bias
                probs = self.softmax(logits)
                
                # Compute gradients
                dloss_dlogits = probs - t
                dloss_dweights = np.outer(x, dloss_dlogits) + self.reg_lambda * self.weights
                dloss_dbias = dloss_dlogits 
                
                # Update weights and bias
                self.weights -= self.learning_rate * dloss_dweights
                self.bias -= self.learning_rate * dloss_dbias
                
                # Calculate accuracy
                if np.argmax(probs) == np.argmax(t):
                    correct_predictions += 1
            
            # Calculate accuracy for the epoch
            accuracy = correct_predictions / len(X)
            self.accuracy_history.append(accuracy)

        print("Training completed.")

    def softmax(self, a):
        # Check if input is 1D or 2D array
        if len(a.shape) == 1:
            # Single sample: subtract max value for numerical stability
            ea = np.exp(a - np.max(a))
            return ea / np.sum(ea)
        else:
            # Batch of samples: subtract max value for each sample
            ea = np.exp(a - np.max(a, axis=1, keepdims=True))
            return ea / np.sum(ea, axis=1, keepdims=True)

    def plot_accuracy_vs_epoch(self):
        epochs = range(1, self.epochs + 1)
        plt.plot(epochs, self.accuracy_history, marker='o')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    def predict(self, X):
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)

    def save(self, path):
        np.save(path, {"weights": self.weights, "bias": self.bias})
        os.rename(path + '.npy', path)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=True).item()
        model = LogisticRegression()
        model.weights = data["weights"]
        model.bias = data["bias"]
        return model
