# Logistic Regression and Multi-Layer Perceptron Classifiers
## Project Overview
This project implements and compares Logistic Regression and Multi-Layer Perceptron (MLP) classifiers for binary classification tasks. The models are evaluated on two datasets: "water" and "loans".

## Datasets

"water" - A smaller dataset
"loans" - A larger dataset

## Model Implementations
### Logistic Regression

Uses gradient descent and matrix inversion
Implements regularization
Preprocessing: Feature rescaling to (0, 1) range

### Multi-Layer Perceptron (MLP)

Architecture: Input layer, hidden layer (10 units with tanh activation), output layer
Stochastic Gradient Descent (SGD) for training
Softmax activation in the output layer
Preprocessing: Feature rescaling to (0, 1) range
Regularization implemented


## Weight Initialization Study
The project includes a comparison of different weight initialization schemes for neural networks:

1. Random initialization
2. Xavier/Glorot initialization
3. He initialization
