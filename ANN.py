from numpy import array, zeros, ones, arange, exp, dot, save, pi
from numpy.random import randn, randint
from numpy.linalg import norm
import matplotlib.pylab as plt

class GeneralNetwork:
    list_of_weight_matrices = []
    list_of_bias_vectors    = []
    def __init__(self, number_of_layers, neurons_per_layer):

        self.list_of_weight_matrices = list_of_weight_matrices
        self.list_of_bias_vectors    = list_of_bias_vectors
        for i in arange(len(neurons_per_layer) - 1):
            self.list_of_weights.append(randn(neurons_per_layer[i],\
                                        neurons_per_layer[i + 1]))
            self.list_of_bias_vectors.append(randn(neurons_per_layer[i], 1))
        





class Network:
    def __init__(self, W2 = 0.5 * randn(2, 2), W3 = 0.5 * randn(3, 2), \
                 W4 = 0.5 * randn(2, 3), b2 = 0.5 * randn(2, 1),       \
                 b3 = 0.5 * randn(3, 1), b4 = 0.5 * randn(2, 1)):

        self.W2 = W2
        self.W3 = W3
        self.W4 = W4

        self.b2 = b2
        self.b3 = b3
        self.b4 = b4

        self.y           = zeros((2, 10))
        self.y[0:1, 0:5] = ones((1,  5))
        self.y[0:1, 5: ] = zeros((1, 5))
        self.y[0:1, 5: ] = zeros((1, 5))
        self.y[1: , 5: ] = ones((1,  5))

        print(self.y)
        self.yt = zeros((2, 1))
        self.x  = zeros((2, 1))

    def train(self, x1, x2, eta = 0.5, niter = 10000):
        total_cost = zeros(niter)
        for counter in arange(niter):
            k          = randint(10)
            self.x[0], self.x[1] = x1[k], x2[k]

            # FORWARD PROP
            a2 = self.activate(self.x,  self.W2,  self.b2)
            a3 = self.activate(a2,      self.W3,  self.b3)
            a4 = self.activate(a3,      self.W4,  self.b4)


            # BACKWARD PROP
            self.yt[0, 0] = self.y[0, k]
            self.yt[1, 0] = self.y[1, k]
            delta4 = a4 * (1 - a4) * (a4 - self.yt)
            delta3 = a3 * (1 - a3) * dot(self.W4.T, delta4)
            delta2 = a2 * (1 - a2) * dot(self.W3.T, delta3)

            # GRADIENT STEP
            self.W2 = self.W2 - eta * delta2 * self.x.T
            self.W3 = self.W3 - eta * delta3 * a2.T
            self.W4 = self.W4 - eta * delta4 * a3.T

            self.b2 = self.b2 - eta * delta2
            self.b3 = self.b3 - eta * delta3
            self.b4 = self.b4 - eta * delta4

            Cost = self.cost(x1, x2)
            self.visual(x1, x2)

            total_cost[counter] = Cost
        return total_cost

    def activate(self, x, W, b):
        return 1 / (1 + exp(-(dot(W, x) + b)))
    def cost(self, x1, x2):

        costvec = zeros((10, 1))
        x       = zeros((2,  1))
        for i in arange(costvec.shape[0]):
            x[0,0], x[1,0] = x1[i], x2[i]
            a2 = self.activate(x,  self.W2, self.b2)
            a3 = self.activate(a2, self.W3, self.b3)
            a4 = self.activate(a3, self.W4, self.b4)
            costvec[i] = norm(self.yt - a4, 2)

        return norm(costvec, 2)**2

    def visual(self, x1, x2):
        plt.scatter(x1[0 : 5], x2[0 : 5], marker = '^',lw = 5, s = pi * 3,\
                                          alpha = 0.5)
        plt.scatter(x1[5 :  ], x2[5 :  ], marker = 'o',lw = 5, s = pi * 3,\
                                          alpha = 1)

        plt.show()
        plt.close()
        return


if __name__ == '__main__':

    x1 = array([0.1, 0.3, 0.1, 0.6, 0.4, \
                0.6, 0.5, 0.9, 0.4, 0.7])
    x2 = array([0.1, 0.4, 0.5, 0.9, 0.2, \
                0.3, 0.6, 0.2, 0.4, 0.6])

    Network = Network()
    #Visual  = Network.visual(x1, x2)
    Cost    = Network.train(x1, x2)
