import numpy as np
import b_logistic_regression.logistic_utils as logistic

N = 100  # Number of samples
D = 2  # Number of features

X = np.random.randn(N,D)
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = np.dot(Xb, w)

result = logistic.sigmoid(z)
print(result)
print(result.shape)
