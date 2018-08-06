import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# Read csv and populate arrays
for line in open('csv/data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Converting regular arrays into numpy arrays
X = np.asarray(X)
Y = np.asarray(Y)

# Visualize the input data
plt.scatter(X, Y)
# plt.show()

# Calculate the best fit a + b
denominator = X.dot(X) - X.mean() * X.sum()

a = (Y.dot(X).sum() - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * Y.dot(X).sum()) / denominator

# Calculate prediction
Y_hat = a * X + b

# Plot with predictions
plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.show()

# Calculate R squared
parenthesis_residual = Y - Y_hat
parenthesis_total = Y - Y.mean()
r_squared = 1 - parenthesis_residual.dot(parenthesis_residual).sum() \
            / parenthesis_total.dot(parenthesis_total).sum()
print(r_squared)
