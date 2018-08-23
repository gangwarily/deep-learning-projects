import numpy as np
import b_logistic_regression.logistic_utils as logistic
import c_deep_learning_1.deep_learning_1_utils as dl1

X, Y = logistic.get_data()
N, D = X.shape

# Size in hidden layer
M = 5
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)

W2 = np.random.randn(M, K)
b2 = np.zeros(K)

P_Y = dl1.forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y, axis=1)

print('Score = ', logistic.score(Y, predictions))

