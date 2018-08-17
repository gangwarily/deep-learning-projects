import numpy as np
from sklearn.utils import shuffle

import b_logistic_regression.logistic_utils as logistic

X, Y = logistic.get_binary_data()
X, Y = shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]

N, D = X.shape
W = np.random.randn(D)
b = 0

# Keep track of the costs across epochs
train_costs = []
test_consts = []
learning_rate = 0.001

for i in range(10000):
    predictionTrain = logistic.forward(Xtrain, W, b)
    predictionTest = logistic.forward(Xtest, W, b)

    c_train = logistic.cross_entropy(Ytrain, predictionTrain)
    c_test = logistic.cross_entropy(Ytest, predictionTest)

    train_costs.append(c_train)
    test_consts.append(c_test)

    W -= learning_rate * Xtrain.T.dot(predictionTrain - Ytrain)
    b -= learning_rate * (predictionTrain - Ytrain).sum()

    if i % 1000 == 0:
        print(i, c_train, c_test)

print('Final train classification_rate', logistic.score(Ytrain, np.round(predictionTrain)))
print('Final test classification_rate', logistic.score(Ytest, np.round(predictionTest)))
