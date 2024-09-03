import sys
import time
import numpy as np
from util import *
from mlp import MultilayerPerceptron
from logreg import LogisticRegression

def main():
    if len(sys.argv) < 3:
        print("Give a datapath for training set as the first command-line argument and validation set as the second.")
        sys.exit(1)

    threshold = 0.6  # Must get this score to pass

    training_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]

    # Load training set
    X_train, T_train = loadDataset(training_data_path)
    T_train = toHotEncoding(T_train, np.max(T_train).astype(int) + 1)

    # Load validation set
    X_val, T_val = loadDataset(validation_data_path)
    T_val = toHotEncoding(T_val, np.max(T_val).astype(int) + 1)

    model = LogisticRegression()
    start_time = time.time()  # Start time measurement
    model.fit(X_train, T_train)
    end_time = time.time()  # End time measurement
    time_taken = end_time - start_time  # Calculate time taken
    print(f'Logreg training time: {time_taken:.4f} seconds')

    # Calculate accuracy on training set
    Y_train = model.predict(X_train)
    acc_train = accuracy(Y_train, T_train)
    print(f'Logistic Regression final accuracy on training set: {acc_train:.4f}')

    # Calculate accuracy on validation set
    Y_val = model.predict(X_val)
    acc_val = accuracy(Y_val, T_val)
    print(f'Logistic Regression accuracy on validation set: {acc_val:.4f}')
    print("")
    # Checking that save/load works.
    model.save('logreg.model')
    model2 = LogisticRegression.load('logreg.model')
    Y_train = model.predict(X_train)
    acc2 = accuracy(Y_train, T_train)
    if abs(acc_train - acc2) > threshold:
        print('WARNING: loaded model performance mismatch.')

    model = MultilayerPerceptron()

    start_time = time.time()  # Start time measurement
    model.fit(X_train, T_train, max_iters=1000)
    end_time = time.time()  # End time measurement
    time_taken = end_time - start_time  # Calculate time taken
    print(f'MLP training time: {time_taken:.4f} seconds')

    # Calculate accuracy on training set
    Y_train = model.predict(X_train)
    acc_train = accuracy(Y_train, T_train)
    print(f'MLP final accuracy on training set: {acc_train:.4f}')

    # Calculate accuracy on validation set
    Y_val = model.predict(X_val)
    acc_val = accuracy(Y_val, T_val)
    print(f'MLP accuracy on validation set: {acc_val:.4f}')

    model.save('mlp.model')

    # Checking that save/load works.
    model.save('mlp.model')
    model2 = MultilayerPerceptron.load('mlp.model')
    Y_train = model.predict(X_train)
    acc2 = accuracy(Y_train, T_train)
    if abs(acc_train - acc2) > threshold:
        print('WARNING: loaded model performance mismatch.')

if __name__ == '__main__':
    main()
