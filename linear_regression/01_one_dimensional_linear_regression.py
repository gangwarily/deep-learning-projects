import numpy as np
import matplotlib.pyplot as plt
import linear_regression.linear_utils as linear

X = []
Y = []

# Read csv and populate arrays
for line in open('csv/data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Converting regular arrays into numpy arrays
X = np.array(X)
Y = np.array(Y)

# Visualize the input data
plt.scatter(X, Y)
# plt.show()

Y_hat = linear.calculate_y_hat(X, Y)

# Plot with predictions
plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.show()

# Calculate R squared
print(linear.calculate_r_squared(Y, Y_hat))
