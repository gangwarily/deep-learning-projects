import numpy as np

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(np.dot(X, W1) + b1)
    y_hat = softmax(Z.dot(W2) + b2)
    return y_hat

