import numpy as np
import matplotlib.pyplot as plt
import linear_regression.linear_utils as linear

BIAS = 1

X = []
Y = []

for line in open('csv/data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), BIAS]) # add the bias term
    Y.append(float(y))


X = np.array(X)
Y = np.array(Y)

print('Shape of X is ' + str(X.shape))
print('Shape of Y is ' + str(Y.shape))

# Plot in 3D coordinate space
# TODO: Code example didn't work...need to investigate

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

print('Shape of w is ' + str(w.shape))

Y_hat = np.dot(X, w)
print('Shape of Y Hat is ' + str(Y_hat.shape))

r_squared = linear.calculate_r_squared(Y, Y_hat)

residual = Y - Y_hat
some = 1 - residual.dot(residual)

print('r_squared is ' + str(r_squared))