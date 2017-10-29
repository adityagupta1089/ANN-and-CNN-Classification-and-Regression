import numpy
import cv2
import random
import math
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
        for i in tqdm(range(1,len(content))):
            line = content[i]
            [name, deg] = line.split('\t')
            # name is of the form './abc' so skip the '.' i.e. name[1:]
            img = cv2.imread('../steering' + name[1:], cv2.IMREAD_GRAYSCALE)
            self.X.append(img.flatten().T / 255)
            self.Y.append(float(deg[-2]))

        print('Read Images')
        # initialize weights and layers
        # ni+1 for a bias term
        self.weights = [(numpy.random.rand(ni_1, ni + 1) - 0.5) / 50 for (ni, ni_1) in zip(architecture, architecture[1:])]

    def split_data(self, ratio):
        split        = math.floor(ratio * len(self.X))
        self.X_train = self.X[0:split]
        self.Y_train = self.Y[0:split]
        self.X_test  = self.X[split + 1:]
        self.Y_test  = self.Y[split + 1:]

    def train(self, epochs, eta, minibatch_size):        
        train_data = list(zip(self.X_train, self.Y_train))
        random.shuffle(train_data)
        N1 = len(train_data)
        K  = len(self.architecture)
        for e in range(epochs):
            print('Epoch #' + str(e+1))
            for i in range(0, N1, minibatch_size):
                train_mini = train_data[i:i + minibatch_size]
                delta_ws = [numpy.zeros(weight.shape) for weight in self.weights]
                for x, y in train_mini:
                    # forward pass
                    vs = [numpy.append([1], x)]                  
                    for i in range(1, K):
                        vs.append(sigmoid(numpy.matmul(self.weights[i-1], vs[-1])))
                        if i != K - 1:
                            vs[-1] = numpy.append([1], vs[-1])
                    # calculation of gradient
                    delta = (vs[-1] - y) * vs[-1] * (1.0 - vs[-1])
                    delta_ws[-1] += eta * delta * vs[-2].T                
                    # backward propagation
                for i in range(K-3, -1, -1):
                    delta = numpy.matmul(self.weights[i+1].T, delta if i == K - 3 else delta[1:])
                    delta_ws[i] += eta * numpy.outer(delta[1:], vs[i])
                # weight update
                self.weights = [weight - delta_w for (weight, delta_w) in zip(self.weights, delta_ws)]       
                    
    def test(self):
        tot_err = 0
        K  = len(self.architecture)
        for x, y in zip(self.X_test, self.Y_test):
            v = numpy.append([1], x)
            for i in range(1, K):
                v = sigmoid(numpy.matmul(self.weights[i-1], v))
                if i != K - 1:
                    v = numpy.append([1], v)
            err = numpy.linalg.norm(y - v) / 2.0
            tot_err += err
        print('Error is '+ str(tot_err))

def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

if __name__ == "__main__":
    
    # create network
    architecture    = [1024, 512, 64, 1]
    split_ratio     = 0.8
    epochs          = 5000
    learning_rate   = 0.01
    minibatch_size  = 64

    network = Neural_Network(architecture)

    network.split_data(split_ratio)
    step = 50
    for _ in range(step):
        network.train(int(epochs / step), learning_rate, minibatch_size)
        network.test()
