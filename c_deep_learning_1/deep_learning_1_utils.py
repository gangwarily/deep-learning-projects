import numpy as np

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    y_hat, Z = forward_backprop(X, W1, b1, W2, b2)
    return y_hat


def forward_backprop(X, W1, b1, W2, b2):
    Z = np.tanh(np.dot(X, W1) + b1)
    y_hat = softmax(Z.dot(W2) + b2)
    return y_hat, Z


def one_hot_Y(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def cross_entropy(T, Y):
    return -np.mean(T*np.log(Y))