# provided by Lazyprogrammer.me
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import numpy as np
import matplotlib.pyplot as plt
import a_linear_regression.linear_utils as linear
import pandas as pd


def get_r_squared(X, Y):
    w = linear.calculate_W(X, Y)
    Y_hat = np.dot(X, w)
    return linear.calculate_r_squared(Y, Y_hat)


excel = pd.read_excel('excel/mlr02.xls')
X = excel.values  # as_matrix() will be deprecated

# Plot X2 to X0
plt.scatter(X[:, 1], X[:, 0])
plt.show()

# Plot X3 to X0
plt.scatter(X[:, 2], X[:, 0])
plt.show()

# Construct necessary data sets
excel['ones'] = 1
Y = excel['X1']
X = excel[['X2', 'X3', 'ones']]
X2only = excel[['X2', 'ones']]
X3only = excel[['X3', 'ones']]

# Calculate and print results
print('r^2 for x2 only =', get_r_squared(X2only, Y))
print('r^2 for x3 only =', get_r_squared(X3only, Y))
print('r^2 for X only =', get_r_squared(X, Y))

