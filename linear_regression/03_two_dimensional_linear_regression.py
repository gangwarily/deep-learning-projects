import numpy as np
import linear_regression.linear_utils as linear

num_features = 2
X = []
Y = []

for line in open('csv/data_2d.csv'):
    line = line.split(',')
    features = line[:num_features]
    y = line[num_features:]

    X.append(features)
    Y.append(y)

X = np.array(X).astype(np.float)
Y = np.array(Y).astype(np.float)

print('Shape of X is ' + str(X.shape))
print('Shape of Y is ' + str(Y.shape))

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

print('Shape of w is ' + str(w.shape))

y_hat = np.dot(X, w)
print('Shape of Y Hat is ' + str(y_hat.shape))

r_squared = linear.calculate_r_squared_matrix(Y, y_hat)


