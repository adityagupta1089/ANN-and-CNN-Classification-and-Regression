import cv2
import datetime
from math import floor
from numpy.lib.stride_tricks import as_strided
import random
from scipy.special import expit
from tqdm import *

import numpy as np


def read_train_data(desc_loc, ims_loc):
    content = []
    with open(desc_loc, 'r') as f:
        content = f.readlines()
    content = content[1:]
    print('Reading train images')
    i = 0
    total = len(content)
    X = []
    Y = []
    for i in tqdm(range(total)):
        line = content[i]
        [name, deg] = line.split('\t')
        img = cv2.imread(ims_loc + name[1:], cv2.IMREAD_GRAYSCALE)
        X.append(img[:,:,np.newaxis] / 255)
        Y.append(float(deg[:-1]))
    return (X, Y)
        
def read_test_data(desc_loc, ims_loc):
    content = []
    with open(desc_loc, 'r') as f:
        content = f.readlines()
    print('Reading test images')
    i = 0
    total = len(content)
    X = []
    for i in tqdm(range(total)):
        line = content[i]
        img = cv2.imread(ims_loc + line[1:-1], cv2.IMREAD_GRAYSCALE)
        X.append(img[:,:,np.newaxis] / 255)
    return X
    
def split_data(X, Y, ratio):
    split = floor(ratio * len(X))
    X_train = np.array(X[:split])
    X_val = np.array(X[split:])
    Y_train = np.array(Y[:split])
    Y_val = np.array(Y[split:])
    return (X_train, Y_train, X_val, Y_val)

def ims2col(imgs, sz, out, stride=1, independent=False):
    i, _, _, d = imgs.shape
    s, s0, s1, s2 = imgs.strides
    k1, k2, _ = sz
    o1, o2, _ = out
    shape = d, k1, k2, o1, o2, i
    strides = s2, stride * s0, stride * s1, s0, s1, s
    out = as_strided(imgs, shape=shape, strides=strides)
    if independent:
        return out.reshape(d, k1 * k2, o1 * o2, i).T
    else:
        return out.reshape(d * k1 * k2, o1 * o2, i).T
    
def col2ims(cols, out):
    o1, o2, d = out
    return cols.reshape(cols.shape[0], o1, o2, d)
    

class Conv:
    def __init__(self, filters, filter_size):
        self.num_filters = filters
        self.filter_size = filter_size

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.h, self.w, self.d = input_shape
        self.p, self.q = self.filter_size
        self.filter_size = (self.p, self.q, self.d)
        self.filters = np.random.randn(self.num_filters, self.p * self.q * self.d) * 0.01
        self.bias = np.random.randn(self.num_filters) * 0.01
        self.output_shape = (self.h - self.p + 1, self.w - self.q + 1, self.num_filters)
        self.padded_shape = (self.h + self.p - 1, self.w + self.q - 1, self.num_filters)
        return self.output_shape
    
    #@profile
    def forward_pass(self, x):
        x_ = ims2col(x, self.filter_size, self.output_shape)
        y_ = x_.dot(self.filters.T) + self.bias
        np.maximum(y_, 0, y_)
        return y_.reshape(x.shape[0], *self.output_shape)     
    
    #@profile
    def grad(self, zs, grad):
        grad_ = np.zeros((zs.shape[0], *self.padded_shape))        
        grad_[:,self.p-1:self.h,self.q-1:self.w,:] = grad
        grad_ = ims2col(grad_, (self.q, self.p, self.num_filters), self.input_shape)
        y_ = grad_.dot(self.filters.reshape(self.d, -1).T)
        y_ = y_.reshape(*zs.shape)
        y_[zs < 0] = 0
        return y_
    
    #@profile
    def update_weights(self, zs, grad, eta):
        zs_ = zs.T
        zs_ = ims2col(zs_, grad.shape[:-1][::-1], self.filter_size)
        grad_ = grad.reshape(-1, self.num_filters)
        y_ = zs_.dot(grad_)
        y_ = y_.T.reshape(self.filters.shape)
        self.filters -= eta * y_
        self.bias -= eta * sum(grad_)
        
class MaxPool:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride
    
    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        h, w, d = input_shape
        p, q = self.size
        self.size = (p, q, d)
        self.output_shape = (int((h - p)/self.stride) + 1, int((w - q)/self.stride) + 1, d)
        o1, o2, _ = self.output_shape
        self.l = o1 * o2
        self.d = d
        self.intermediate_shape = (o1 * o2, p * q, d)
        return self.output_shape
    
    def forward_pass(self, x):
        x_ = ims2col(x, self.size, self.output_shape, stride=self.stride, independent=True)
        self.max_indices = np.argmax(x_, axis=-2)
        max_indices_indices = np.indices(self.max_indices.shape)
        self.a_, self.b_, self.c_ = max_indices_indices
        return x_[self.a_, self.b_, self.max_indices, self.c_].reshape(x.shape[0], *self.output_shape)  
            
    def update_weights(self, zs, grad, eta):
        pass
    
    def grad(self, zs, grad):
        grad = grad.reshape(zs.shape[0], self.l, self.d)
        grad_ = np.zeros((zs.shape[0], *self.intermediate_shape))
        grad_[self.a_, self.b_, self.max_indices, self.c_] = grad
        return col2ims(grad_, self.input_shape)
        
class Dense:
    def __init__(self, nodes, activation):
        self.nodes = nodes
        self.sigmoid = activation == 'sigmoid'
        self.linear = activation == 'linear'
    
    def set_input_shape(self, input_shape):
        n, = input_shape
        self.weight = np.random.randn(n, self.nodes)
        self.bias = np.random.randn(self.nodes)
        return (self.nodes,)
    
    def forward_pass(self, x):
        out = x.dot(self.weight) + self.bias
        if self.sigmoid:
            return expit(out)
        elif self.linear:
            return out

    def grad(self, zs, grad):
        grad = grad.dot(self.weight.T)
        if self.sigmoid:
            grad *= zs * (1 - zs) 
        elif self.linear:
            pass
        return grad
        
    def update_weights(self, zs, grad, eta):
        self.weight -= eta * zs.T.dot(grad)
        self.bias -= eta * sum(grad)
        
    
class Flatten:
    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        h, w, d = input_shape
        self.output_shape = (h * w * d,)
        return self.output_shape
    
    def forward_pass(self, x):
        return x.reshape(x.shape[0], *self.output_shape)
    
    def grad(self, zs, grad):
        return grad.reshape(zs.shape)
    
    def update_weights(self, zs, grad, eta):
        pass
      
      
def calculate_error(architecture, X_val, Y_val):
    N = len(X_val)
    zs = np.array(X_val)
    for layer in architecture:
        zs = layer.forward_pass(zs)
    error = np.square(np.linalg.norm(zs - np.atleast_2d(Y_val).T)) / 2
    return error / N
  
def train_network(architecture, X_train, Y_train, X_val, Y_val, \
        minibatch_size, eta, X_test):
    N = len(X_train)
    
    order = list(range(N))
    random.shuffle(order)
    
    X_train = X_train[order]
    Y_train = Y_train[order]
    
    minibatches = [(X_train[i:i+minibatch_size], np.atleast_2d(np.array(Y_train[i:i+minibatch_size])).T)
        for i in range(0, N, minibatch_size)]
    
    for e in range(epochs):
        train_error = 0
        for xs, ys in tqdm(minibatches):
            zss = [xs]
            for layer in architecture:
                zss.append(layer.forward_pass(zss[-1]))
            train_error += np.square(np.linalg.norm(zss[-1] - ys)) / 2
            grad = zss[-1] - ys
            for layer, zs in zip(reversed(architecture),reversed(zss[:-1])):
                layer.update_weights(zs, grad / xs.shape[0], eta)
                grad = layer.grad(zs, grad)
        train_error /= N
        validation_error = calculate_error(architecture, X_val, Y_val)
        print('Epoch %04d, Train Error: %f, Validation Error: %f' % (e + 1, train_error, validation_error))
        if (e+1) % 10 == 0:
            time = datetime.datetime.now().strftime("%d_%m_%H:%M")
            write_results(architecture, X_test, 'results_' + str(time) + '.txt')
            print('Wrote result to "results_%s.txt"' % time)
        
def write_results(architecture, X_test, file):
    N = len(X_test)
    zs = np.array(X_test)
    for layer in architecture:
        zs = layer.forward_pass(zs)
    with open(file, 'w') as f:
        for z in zs:
            f.write(str(z[0])+'\n')

if __name__ == "__main__":
    split_ratio = 0.8
    epochs = 5000
    minibatch_size = 64
    eta = 0.001

    X, Y = read_train_data('../steering/data.txt', '../steering')
    X_test = read_test_data('./test/test-data.txt', './test')

    X_train, Y_train, X_validation, Y_validation = split_data(X, Y, split_ratio)
    
    architecture = [
            Conv(4, (3,3)),
            MaxPool((2,2), 2),
            Flatten(),
            Dense(256, 'sigmoid'),
            Dense(64, 'sigmoid'),
            Dense(64, 'sigmoid'),
            Dense(1, 'linear'),
        ]
        
    input_shape = X[0].shape
    for layer in architecture:
        input_shape = layer.set_input_shape(input_shape)
    
    train_network(architecture, X_train, Y_train, X_validation, Y_validation, \
        minibatch_size, eta, X_test)
