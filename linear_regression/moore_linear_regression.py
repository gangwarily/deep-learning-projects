import re
import numpy as np
import matplotlib.pyplot as plt
import linear_regression.linear_utils as linear

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('csv/moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))

    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

# Show plot of untransformed data
plt.scatter(X, Y)
plt.show()

# Turn plot into linear with log
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

Y_hat = linear.calculate_y_hat(X, Y)

plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.show()

r_squared = linear.calculate_r_squared(Y, Y_hat)
print(r_squared)


