# Make sure to download the kaggle csv file for digit recognizer
import theano.tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt

from d_deep_learning_2.util import get_normalized_data, y2indicator, error_rate

Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

max_iter = 20
print_period = 10

lr = 0.0004
reg = 0.01

Xtrain = Xtest.astype(np.float32)
Ytrain = Ytest.astype(np.float32)
Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
Ytest_ind = y2indicator(Ytest).astype(np.float32)

N, D = Xtrain.shape
batch_sz = 500
n_batches = N // batch_sz

M = 300
K = 10
W1_init = np.random.randn(D, M) / 28
b1_init = np.zeros(M)
W2_init = np.random.randn(M, K) / np.sqrt(M)
b2_init = np.zeros(K)

thX = T.matrix('X')
thT = T.matrix('T')
W1 = theano.shared(W1_init, 'W1')
b1 = theano.shared(b1_init, 'b1')
W2 = theano.shared(W2_init, 'W2')
b2 = theano.shared(b2_init, 'b2')

thZ = T.nnet.relu(thX.dot(W1) + b1)
thY = T.nnet.softmax(thZ.dot(W2) + b2)

cost = -(thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
# cost = T.sum(T.nnet.categorical_crossentropy(thY, thT)) # Messing with using theano's categorical cross-entropy func

prediction = T.argmax(thY, axis=1)

update_W1 = W1 - lr * T.grad(cost, W1)
update_b1 = b1 - lr * T.grad(cost, b1)
update_W2 = W2 - lr * T.grad(cost, W2)
update_b2 = b2 - lr * T.grad(cost, b2)

train = theano.function(
    inputs=[thX, thT],
    updates=[(W1, update_W1), (W2, update_W2), (b1, update_b1), (b2, update_b2)]
)

get_prediction = theano.function(
    inputs=[thX, thT],
    outputs=[cost, prediction]
)

costs = []
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
        Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

        train(Xbatch, Ybatch)
        if j % print_period == 0:
            cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
            err = error_rate(prediction_val, Ytest)
            print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
            costs.append(cost_val)

plt.plot(costs)
plt.show()