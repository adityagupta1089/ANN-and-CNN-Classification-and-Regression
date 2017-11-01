import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import *
from scipy.special import expit
import datetime

class Neural_Network:
    # data lists
    X = []
    Y = []
    # training data lists
    X_train = []
    Y_train = []
    # testing data lists
    X_test = []
    Y_test = []

    def __init__(self, architecture):
        self.architecture = architecture

        # read details file
        content = []
        with open('../steering/data.txt', 'r') as f:
            content = f.readlines()
        # first image is non-existent
        content = content[1:]

        # read individual images
        print('Reading images')
        i=0
        total = len(content)
        for i in tqdm(range(total)):
            line = content[i]
            [name, deg] = line.split('\t')
            # name is of the form './abc' so skip the '.' i.e. name[1:]
            img = cv2.imread('../steering' + name[1:], cv2.IMREAD_GRAYSCALE)
            self.X.append(img.flatten().T / 255)
            self.Y.append(float(deg[:-1]))

        # initialize weights and layers
        # ni+1 for a bias term
        self.weights = [(np.random.rand(ni_1, ni + 1) - 0.5) / 50 for (ni, ni_1) in zip(architecture, architecture[1:])]

    def split_data(self, ratio):
        split        = math.floor(ratio * len(self.X))
        self.X_train = np.column_stack(tuple(self.X[0:split]))
        self.Y_train = np.array(self.Y[0:split])
        self.X_test  = np.column_stack(tuple(self.X[split + 1:]))
        self.Y_test  = np.array(self.Y[split + 1:])

    def train(self, epochs, eta, minibatch_size, log_file):
        # constants
        N = self.X_train.shape[1]
        K = len(self.architecture)

        # shuffling
        order = list(range(N))
        random.shuffle(order)
        self.X_train = self.X_train[:,order]
        self.Y_train = self.Y_train[order]

        # error containers
        train_errors = [self.test(train=True)]
        test_errors  = [self.test(train=False)]

        # epochwise training
        print('Training for %d epochs' % epochs)
        for e in tqdm(range(epochs)):
            # processing each batch
            for i in range(0, N, minibatch_size):
                # obtain batch
                x = self.X_train[:,i:i+minibatch_size]
                y = self.Y_train[i:i+minibatch_size]
                _N = x.shape[1]
                delta_ws = [np.zeros(weight.shape) for weight in self.weights]
                # forward pass
                Vs = [np.row_stack((np.ones((1,_N)), x))]
                for i in range(1, K):
                    if i != K - 1:
                        # V(i) = sigma (w(i-1) * V(i-1))
                        Vs.append(sigmoid(self.weights[i-1].dot(Vs[-1])))
                        # add bias terms
                        Vs[-1] = np.row_stack((np.ones((1,_N)), Vs[-1]))
                    else:
                        # V(k) = w(k-1) * V(k-1), no bias term
                        Vs.append(self.weights[i-1].dot(Vs[-1]))
                # calculation of gradient
                # Xi(k) = V(k) - O(k)
                xi = Vs[-1] - y
                # Delta w(k-1) = eta * Xi(k) * V(k-1)^T
                delta_ws[-1] = eta * xi.dot(Vs[-2].T)
                # backward propagation
                for i in range(K-3,-1,-1):
                    # Xi(i+1) = V(i+1) * (1 - V(i+1)) * (w(i+1)^T * Xi(i+2))
                    xi = Vs[i+1] * (1 - Vs[i+1]) * self.weights[i+1].T.dot(xi)
                    # trimmed first row of Xi to prevent flow of gradients from biases
                    xi = xi[1:,:]
                    # Delta w(i) = eta * Xi(i+1) * Vs(i)^T
                    delta_ws[i] = eta * xi.dot(Vs[i].T)
                    
                # weight update
                self.weights = [weight - delta_w for (weight, delta_w) in zip(self.weights, delta_ws)]

            # calculation of error
            train_error = self.test(train=True)
            test_error  = self.test(train=False)
            train_errors.append(train_error)
            test_errors.append(test_error)
            log_file.write('%d\tTrain:%f\tTest:%f\n' % (e+1, train_error, test_error))
            if e % 100 == 0:
                log_file.flush()
        return (train_errors, test_errors)

    def test(self, train):
        # check test/training
        if train:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        # constants
        K  = len(self.architecture)
        N = x.shape[1]

        # forward pass
        Vs = np.row_stack((np.ones((1,N)), x))
        for i in range(1, K):
            if i != K - 1:
                Vs = sigmoid(self.weights[i-1].dot(Vs))
                Vs = np.row_stack((np.ones((1,N)), Vs))
            else:
                Vs = self.weights[i-1].dot(Vs)

        # sum of squared error
        err = np.square(np.linalg.norm(Vs - y))
        return err / N

def sigmoid(z):
    return expit(z)

if __name__ == "__main__":

    # create network
    architecture    = [1024, 512, 64, 1]
    split_ratio     = 0.8
    epochs          = 1000
    learning_rate   = 0.01
    minibatch_size  = 32

    # neural network training
    network = Neural_Network(architecture)
    network.split_data(split_ratio)
    time = datetime.datetime.now().strftime("%d_%m_%H:%M")
    f = open('log_'+ time +'.log', 'w')
    f.write('architecture = ' + str(architecture) + '\n')
    f.write('split_ratio = ' + str(split_ratio) + '\n')
    f.write('epochs = ' + str(epochs) + '\n')
    f.write('learning_rate = ' + str(learning_rate) + '\n')
    f.write('minibatch_size = ' + str(minibatch_size) + '\n')
    f.flush()
    train_errors, test_errors = network.train(epochs, learning_rate, minibatch_size, f)
    f.close()
    for i in range(len(network.weights)):
        np.savetxt('weight_' + time + '_' + str(i), network.weights[i])

    # plotting
    plt.plot(np.arange(0,epochs+1), train_errors, label='Training Error')
    plt.plot(np.arange(0,epochs+1), test_errors, label='Testing Error')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.xlim([0, epochs])
    plt.legend(loc=0)

    # saving figure
    plt.savefig('plot_' + time + '.eps')
    plt.savefig('plot_' + time + '.png')

    # show plot
    plt.show()
