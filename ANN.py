from numpy import array, zeros, ones, arange, exp, dot, save, pi, linspace,\
                  matrix, ceil
from numpy.random import randn, randint, uniform, normal
from numpy.linalg import norm
import matplotlib.pylab as plt
import sys

class GeneralNetwork:
    def __init__(self, number_of_layers, neurons_per_layer):

        try:
            if number_of_layers != len(neurons_per_layer):
                print("The length of the neuron vector has to match the number \
                        of layers. Each entry refers to the number of neurons in\
                        that layer!")
                sys.exit()
        except:
            print("The variable number_of_layers must be an integer!")
            print("The variable neurons_per_layer must be a list!")
            sys.exit()

        list_of_weight_matrices = [normal(size=(neurons_per_layer[0], \
                                                neurons_per_layer[0]))]
        list_of_bias_vectors    = [normal(size=(neurons_per_layer[0], 1))]

        self.weights            = list_of_weight_matrices
        self.biases             = list_of_bias_vectors

        self.number_of_layers   = number_of_layers
        self.neurons_per_layer  = neurons_per_layer

        for i in arange(1, len(neurons_per_layer)):
            self.weights.append(normal(size=(neurons_per_layer[i],\
                                        neurons_per_layer[i - 1])))
            self.biases.append(normal(size=(neurons_per_layer[i], 1)))

    def activate(self, x, W, b):

        return 1 / (1 + exp(-(dot(W, x) + b)))

    def train(self, Data, eta = 0.1, niter = 100000):

        cost   = zeros(niter)
        xtrain = zeros((Data.xtrain.shape[0], 1))
        ytrain = zeros((Data.ytrain.shape[0], 1))
        activation = []
        delta      = []
        for counter in arange(niter):
            k = randint(Data.xtrain.shape[1])
            # Training Data
            xtrain[:, 0], ytrain[:, 0] = Data.xtrain[:, k], Data.ytrain[:, k]

            # Forward Prop
            for s in arange(self.number_of_layers):
                activation.append(self.activate(xtrain,            \
                        self.weights[s], self.biases[s]))
                xtrain = activation[s]
            #  Back Prop

        #    delta.append(activation[-1] * (1 - activation[-1]) * (activation[-1] - ytrain))
        #    delta.append(activation[-2] * (1 - activation[-2]) * dot(self.weights[-1].T, delta[0]))
        #    delta.append(activation[-3] * (1 - activation[-3]) * dot(self.weights[-2].T, delta[1]))


            delta.append(activation[-1] * (1 - activation[-1]) * (activation[-1] - ytrain))
            for s in arange(0, self.number_of_layers-1):
                delta.append(activation[-2 - s] * (1 - activation[-2 - s]) * \
                dot(self.weights[-1  - s].T, delta[s]))


            #  Update

            self.weights[0]  -= eta * delta[-1] * Data.xtrain[:, k].T
            for s in arange(1, self.number_of_layers):
                if s >= 1:
                    self.weights[s]   -= eta * dot(delta[-(s + 1)], activation[s - 1].T)

            deltaflip = delta
            deltaflip.reverse()

            for s in arange(self.number_of_layers):
                self.biases[s] -= eta * deltaflip[s]

            activation = []
            delta      = []

            cost[counter] = self.cost_function(Data)

        return cost

    def cost_function(self, Data):
        temp_cost = zeros((Data.xtrain.shape[1],1))
        x         = zeros((Data.xtrain.shape[0],1))

        for i in arange(temp_cost.shape[0]):
            x[0,0], x[1,0] = Data.xtrain[0,i], Data.xtrain[1,i]

    #        a2 = self.activate(x, self.weights[0],  self.biases[0])
    #        a3 = self.activate(a2, self.weights[1], self.biases[1])
    #        a4 = self.activate(a3, self.weights[2], self.biases[2])

            for s in arange(self.number_of_layers):
                a = self.activate(x, self.weights[s], self.biases[s])
                x = a
            temp_cost[i] = norm(x.ravel() - Data.ytrain[:, i], 2)
        return norm(temp_cost, 2)**2
class Data:
    def __init__(self, number_of_data_points,highamdata=True):


        if not highamdata:
            self.x           = linspace(0, 1, number_of_data_points)
            x1               = zeros((1, number_of_data_points))
            x2               = zeros((1, number_of_data_points))
            x1[0,:]          = uniform(0, -self.x + 1, self.x.shape[0])
            x2[0,:]          = uniform(-self.x + 1, 1, self.x.shape[0])
        else:

            x1 = array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
            x2 = array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])


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

    Network = GeneralNetwork(3, [2, 20, 2])
    # define network architecture using GeneralNetwork object
    Data    = Data(10, highamdata = True)
    # create data using Data object
    cost = Network.train(Data)
    # train the network with the data

    import matplotlib.pylab as plt
    # import plotting module
    plt.plot(cost)
    plt.xlabel(r'Iteration Number', fontsize = 24)
    plt.ylabel(r'Cost', fontsize = 24)
    plt.savefig('demonstration.png')
    plt.show()
