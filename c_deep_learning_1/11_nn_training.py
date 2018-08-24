import numpy as np
from sklearn.utils import shuffle
import b_logistic_regression.logistic_utils as logistic
import c_deep_learning_1.deep_learning_1_utils as dl1

X, Y = logistic.get_data()
X, Y = shuffle(X, Y)

M = 5
N, D = X.shape
Y = Y.astype(np.int32)
K = len(set(Y))

X_train = X[:-100]
X_test = X[-100:]
Y_train = Y[:-100]
Y_test = Y[-100:]

Y_train_encoded = dl1.one_hot_Y(Y_train, K)
Y_test_encoded = dl1.one_hot_Y(Y_test, K)

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

learning_rate = 0.001


for i in range(10000):
    p_Y_train, Z_train = dl1.forward_backprop(X_train, W1, b1, W2, b2)
    p_Y_test, Z_test = dl1.forward_backprop(X_test, W1, b1, W2, b2)
    cost_train = dl1.cross_entropy(Y_train_encoded, p_Y_train)
    cost_test = dl1.cross_entropy(Y_test_encoded, p_Y_test)

    W2 -= learning_rate * Z_train.T.dot(p_Y_train - Y_train_encoded)
    b2 -= learning_rate * (p_Y_train - Y_train_encoded).sum(axis=0)
    dZ = (p_Y_train - Y_train_encoded).dot(W2.T) * (1 - Z_train * Z_train)
    W1 -= learning_rate * X_train.T.dot(dZ)
    b1 -= learning_rate * dZ.sum(axis=0)

    if i % 1000 == 0:
        print(i, cost_train, cost_test)

print(p_Y_train[1])
print(np.argmax(p_Y_train, axis=1)[1])
print(Y_train[1])

print("Final train classification_rate:", logistic.score(Y_train, np.argmax(p_Y_train, axis=1)))
print("Final test classification_rate:", logistic.score(Y_test, np.argmax(p_Y_test, axis=1)))






