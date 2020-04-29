import numpy as np
from numpy.random import uniform, randint, randn, normal
import sys
from timeit import default_timer as dt
def activate(x, W, b):
    '''
    Inputs:

    Outputs:

    Description: Sigmoid activation function
    '''
    return 1 / (1 + np.exp(-(np.dot(W, x) + b)))

def cost_function(W2,W3,W4,b2,b3,b4, x1, x2,y):

    '''
    Inputs:

    Outputs:

    Description:
    '''

    costvec = np.zeros((10, 1))
    x       = np.zeros((2,  1))
    for i in np.arange(costvec.shape[0]):
        x[0,0], x[1,0] = x1[i], x2[i]
        a2 = activate(x,  W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        costvec[i] = np.linalg.norm(a4.ravel()-y[:,i], 2)

    return np.linalg.norm(costvec, 2)**2

def predict(W2, W3, W4, b2, b3, b4, xvec):

    a2 = activate(xvec, W2, b2)
    a3 = activate(a2,   W3, b3)
    a4 = activate(a3,   W4, b4)

    return a4

import matplotlib.pylab as plt

x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])

y           = np.zeros((2, 10))
y[0:1, 0:5] = np.ones((1,  5))
y[0:1, 5: ] = np.zeros((1, 5))
y[0:1, 5: ] = np.zeros((1, 5))
y[1: , 5: ] = np.ones((1,  5))

W2,W3,W4 = normal(size=(2,2)), normal(size=(3,2)), normal(size=(2, 3))
b2,b3,b4 = normal(size=(2,1)), normal(size=(3,1)), normal(size=(2, 1))

eta = 0.75
Niter = 100000


xvec = np.zeros((2,1))
yvec = np.zeros((2,1))
cost_value = np.zeros((Niter,1))
for counter in np.arange(Niter):
    k = randint(10)
    xvec[0,0], xvec[1,0] = x1[k], x2[k]
    yvec[:,0] = y[:, k]

    # forward pass
    a2 = activate(xvec, W2, b2)
    a3 = activate(a2,   W3, b3)
    a4 = activate(a3,   W4, b4)

    # backward pass
    delta4 = a4 * (1 - a4) * (a4 - yvec)

    delta3 = a3 * (1 - a3) * np.dot(W4.T, delta4)

    delta2 = a2 * (1 - a2) * np.dot(W3.T, delta3)

    W2 -= eta * delta2 * xvec.T
    W3 -= eta * delta3 * a2.T
    W4 -= eta * delta4 * a3.T

    b2 -= eta * delta2
    b3 -= eta * delta3
    b4 -= eta * delta4

    cost_value[counter] = cost_function(W2,W3,W4,b2,b3,b4,x1, x2, y)

#plt.plot(cost_value)
#plt.show()

X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

X1Test = np.array(X.ravel())
X2Test = np.array(Y.ravel())

XTest  = np.stack((X1Test, X2Test), axis = 1)
empty  = np.zeros(200*200)


for i in np.arange(XTest.shape[0]):

    xvec[0,0], xvec[1,0] = XTest[i, 0], XTest[i, 1]


    YPredictions = predict(W2, W3, W4, b2, b3, b4, xvec)
    YPredictions = np.array(YPredictions[0] >= YPredictions[1])


    if YPredictions[0] == True:
        empty[i] = 1


YPred = empty.reshape((200, 200))
import matplotlib.pyplot as plt
plt.figure()
#plt.imshow(YPred)
plt.contourf(X, Y, YPred)

plt.scatter(x1[0:5], x2[0:5], marker='^', lw=5)
plt.scatter(x1[5:],  x2[5:], marker='o', lw=5)
plt.show()
