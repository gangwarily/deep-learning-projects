import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('csv/ecommerce_data.csv')
    data = df.values

    # Notes on indexing:
    # : means the entire row / column
    # : negative index means counting from the end of the array
    X = data[:, : - 1]
    Y = data[:, -1]

    # Print shapes
    print('The shape of X =', X.shape)
    print('The shape of Y =', Y.shape)

    # Normalize data
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    # Categorical columns (for time of day)
    N, D = X.shape
    X_encoded = np.zeros((N, D + 3))  # We are one hot encoding categorical values (Add 3 columns)
    X_encoded[:, 0:(D - 1)] = X[:, 0:(D - 1)]  # Copy all values except values from the last column

    for n in range(N):
        time = int(X[n, D - 1])  # Get time value from the last column
        X_encoded[n, time + D - 1] = 1  # Update index of for the time with an one hot encode

    return X_encoded, Y


# This function will remove data that is greater than 2 as this class doesn't cover multi-class classification
def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    return X2, Y2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, W, b):
    return sigmoid(np.dot(X, W) + b)


def score(Y, P):
    return np.mean(Y == P)


def cross_entropy(T, Y):
    return - np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y))