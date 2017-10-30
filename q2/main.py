import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import *


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
        for i in tqdm(range(len(content))):
            line = content[i]
            [name, deg] = line.split('\t')
            # name is of the form './abc' so skip the '.' i.e. name[1:]
            img = cv2.imread('../steering' + name[1:], cv2.IMREAD_GRAYSCALE)
            self.X.append(img.flatten().T / 255)
            self.Y.append(float(deg[:-2]))

        # initialize weights and layers
        # ni+1 for a bias term
        self.weights = [(np.random.rand(ni_1, ni + 1) - 0.5) / 50 for (ni, ni_1) in zip(architecture, architecture[1:])]

    def split_data(self, ratio):
        split        = math.floor(ratio * len(self.X))
        self.X_train = np.column_stack(tuple(self.X[0:split]))
        self.Y_train = np.array(self.Y[0:split])
        self.X_test  = np.column_stack(tuple(self.X[split + 1:]))
        self.Y_test  = np.array(self.Y[split + 1:])

    def train(self, epochs, eta, minibatch_size):
        N = self.X_train.shape[1]
        K = len(self.architecture)
        order = list(range(N))
        random.shuffle(order)
        self.X_train = self.X_train[:,order]
        self.Y_train = self.Y_train[order]
        train_errors = []
        test_errors  = []
        print('Training for %d epochs' % epochs)
        for e in tqdm(range(epochs)):
            for i in range(0, N, minibatch_size):
                x = self.X_train[:,i:i+minibatch_size]
                y = self.Y_train[i:i+minibatch_size]
                _N = x.shape[1]
                delta_ws = [np.zeros(weight.shape) for weight in self.weights]
                # forward pass
                Vs = [np.row_stack((np.ones((1,_N)), x))]
                for i in range(1, K):
                    Vs.append(sigmoid(self.weights[i-1].dot(Vs[-1])))
                    if i != K - 1:
                        Vs[-1] = np.row_stack((np.ones((1,_N)), Vs[-1]))
                # calculation of gradient
                xi = (Vs[-1] - y) * Vs[-1] * (1.0 - Vs[-1])
                delta_ws[-1] = eta * xi.dot(Vs[-2].T)
                # backward propagation
                for i in range(K-3,-1,-1):
                    xi = self.weights[i+1].T.dot(xi)
                    delta_ws[i] = eta * xi[1:,:].dot(Vs[i].T)
                    xi = xi[1:,:]
                # weight update
                self.weights = [weight - delta_w for (weight, delta_w) in zip(self.weights, delta_ws)]
            """train_error = self.test(train=True)
            test_error  = self.test(train=False)
            train_errors.append(train_error)
            test_errors.append(test_error)"""
        return (train_errors, test_errors)

    def test(self, train):
        if train:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test
        err = 0
        K  = len(self.architecture)
        N = x.shape[1]
        Vs = [np.row_stack((np.ones((1,N)), x))]
        for i in range(1, K):
            Vs.append(sigmoid(self.weights[i-1].dot(Vs[-1])))
            if i != K - 1:
                Vs[-1] = np.row_stack((np.ones((1,N)), Vs[-1]))
        err = np.linalg.norm(Vs[-1] - y) / 2
        return err

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":

    # create network
    architecture    = [1024, 512, 64, 1]
    split_ratio     = 0.8
    epochs          = 5000
    learning_rate   = 0.01
    minibatch_size  = 64

    network = Neural_Network(architecture)

    network.split_data(split_ratio)
    
    train_errors, test_errors = network.train(epochs, learning_rate, minibatch_size)
    plt.plot(np.arange(1,epochs+1), train_errors, label='Training Error')
    plt.plot(np.arange(1,epochs+1), test_errors, label='Testing Error')
    plt.axis([0,epochs+1,0,3.5])
    plt.ylabel('Sum of Squared Error')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.show()
