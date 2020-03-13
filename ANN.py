from numpy import array, zeros, ones, arange, exp, dot, save, pi, linspace,\
                  matrix, ceil
from numpy.random import randn, randint, uniform
from numpy.linalg import norm
import matplotlib.pylab as plt

class GeneralNetwork:
    def __init__(self, number_of_layers, neurons_per_layer):
        list_of_weight_matrices = [randn(neurons_per_layer[0], neurons_per_layer[0])]
        list_of_bias_vectors    = [randn(neurons_per_layer[0], 1)]

        self.weights = list_of_weight_matrices
        self.biases    = list_of_bias_vectors

        self.number_of_layers   = number_of_layers
        self.neurons_per_layer  = neurons_per_layer

        for i in arange(1, len(neurons_per_layer)):
            self.weights.append(randn(neurons_per_layer[i],\
                                        neurons_per_layer[i - 1]))
            self.biases.append(randn(neurons_per_layer[i], 1))

    def activate(self, x, W, b):
        return 1 / (1 + exp(-(dot(W, x) + b)))

    def train(self, Data, eta = 0.05, niter = 10000):

        cost   = zeros(niter)
        xtrain = zeros((Data.xtrain.shape[0], 1))
        ytrain = zeros((Data.ytrain.shape[0], 1))

        activation = []
        delta      = []
        for counter in arange(niter):
            k = randint(Data.xtrain.shape[0])
            xtrain[:, 0] = Data.xtrain[:, k]

            # Forward Prop
            for s in arange(self.number_of_layers):
                activation.append(self.activate(xtrain,            \
                        self.weights[s], self.biases[s]))
                xtrain = activation[s]
            ytrain[:, 0] = Data.ytrain[:, k]

            # Back Prop

            delta.append(activation[-1] * (1 - activation[-1]) * \
                              activation[-1] - ytrain)
            for s in arange(0, self.number_of_layers-1):
                delta.append(activation[-2 - s] * (1 - activation[-2 - s]) * \
                dot(self.weights[-1  - s].T, delta[s]))

            # Update

            self.weights[0]  -= eta * delta[-1] * xtrain.T

            self.weights[1]  -= eta * delta[-2] * activation[1]
            self.weights[2]  -= eta * delta[-3] * activation[2]
            for s in arange(0, self.number_of_layers):
                self.biases[s]        -= eta * delta[s]
                if s >= 1:
                    self.weights[s]   -= eta * delta[-(s + 1)] * activation[s]


            activation = []
            delta      = []

            #cost[counter] = self.cost_function(Data)
            print(self.cost_function(Data))
        return cost

    def cost_function(self, Data):
        temp_cost = zeros((Data.xtrain.shape[1],1))
        x         = zeros((Data.xtrain.shape[0],1))
        activation = []
        for i in arange(temp_cost.shape[0]):
            x[0,0], x[1,0] = Data.xtrain[0,i], Data.xtrain[1,i]
            a2 = self.activate(x,  self.weights[0], self.biases[0])
            a3 = self.activate(a2,  self.weights[0], self.biases[0])
            a4 = self.activate(a3,  self.weights[0], self.biases[0])

            #for s in arange(self.number_of_layers):
            #    activation.append(self.activate(xtrain,            \
            #            self.weights[s], self.biases[s]))
            #    xtrain = activation[s]
            temp_cost[i] = norm(Data.ytrain - a4, 2)
        return norm(temp_cost, 2)**2

class Data:
    def __init__(self, number_of_data_points):
        self.x           = linspace(0, 1, number_of_data_points)

        x1               = zeros((1, number_of_data_points))
        x2               = zeros((1, number_of_data_points))

        x1[0,:]          = uniform(0, -self.x + 1, self.x.shape[0])
        x2[0,:]          = uniform(-self.x + 1, 1, self.x.shape[0])

        y1               = zeros((1, int(number_of_data_points / 2)))
        y2               = ones((1,  int(number_of_data_points / 2)))

        self.xtrain      = zeros((2, number_of_data_points))
        self.ytrain      = zeros((2, number_of_data_points))

        self.xtrain[0,:], self.xtrain[1,:]                = x1, x2
        self.ytrain[0, 0:int(number_of_data_points / 2)]  = y2
        self.ytrain[0, int(number_of_data_points   / 2):] = y1
        self.ytrain[1, 0:int(number_of_data_points / 2)]  = y1
        self.ytrain[1, int(number_of_data_points   / 2):] = y2

if __name__ == '__main__':
    Network = GeneralNetwork(3, [2, 3, 2])
    Data    = Data(10)
    Network.train(Data)
