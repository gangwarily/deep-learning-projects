import numpy as np
import b_logistic_regression.logistic_utils as logistic

X, Y = logistic.get_binary_data()

N, D = X.shape
W = np.random.randn(D)
b = 0

Y_predict = logistic.forward(X, W, b)
predictions = np.round(Y_predict)

# When using randomly distributed weights, the results are mediocre
print('Score =', logistic.score(Y, predictions))
