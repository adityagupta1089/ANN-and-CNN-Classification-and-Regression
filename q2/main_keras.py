import cv2
import math
import numpy as np
import random
from tqdm import *
import datetime
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import keras

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

    def __init__(self):

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
            self.X.append(img[:,:,np.newaxis] / 255)
            self.Y.append(float(deg[:-1]))
        """ COVNET """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(32,32,1)))
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='sigmoid'))
        self.model.add(Dense(1,activation='linear'))
        
        self.model.compile(loss='mean_squared_error',
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
        
        
        """ DENSE
        self.model = Sequential()
        self.model.add(Dense(512, activation='sigmoid', input_shape=(1024,)))
        self.model.add(Dense(64, activation='sigmoid'))    
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='sgd',
            loss='mean_squared_error')
        """

    def split_data(self, ratio):
        split        = math.floor(ratio * len(self.X))
        self.X_train = np.array(self.X[0:split])
        self.Y_train = np.array(self.Y[0:split])
        self.X_test  = np.array(self.X[split + 1:])
        self.Y_test  = np.array(self.Y[split + 1:])

    def train(self, epochs, eta, minibatch_size):
        self.model.fit(self.X_train, self.Y_train,
            batch_size=minibatch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(self.X_test, self.Y_test))
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)        
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

def sigmoid(z):
    return expit(z)

if __name__ == "__main__":

    # create network
    split_ratio     = 0.8
    epochs          = 1000
    learning_rate   = 0.01
    minibatch_size  = 64

    # neural network training
    network = Neural_Network()
    network.split_data(split_ratio)
    
    network.train(epochs, learning_rate, minibatch_size)
