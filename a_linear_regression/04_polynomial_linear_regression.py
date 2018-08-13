import numpy as np
import matplotlib.pyplot as plt
import a_linear_regression.linear_utils as linear

X = []
Y = []

for line in open('csv/data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x * x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:, 1], Y)
plt.show()

W = linear.calculate_W(X, Y)
Y_hat = np.dot(X, W)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))
plt.show()

r_squared = linear.calculate_r_squared(Y, Y_hat)
print('R^2 is ' + str(r_squared))