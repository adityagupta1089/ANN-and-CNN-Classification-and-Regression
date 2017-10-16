import numpy as np
import math
import cv2

INITIAL_WEIGHT_RANGE = 0.01

class Neural_Network:
    def __init__(self, architecture, training_data, testing_data):
        self.architecture = architecture
        self.training_data = training_data
        self.testing_data = testing_data
        self.weights = []
        for i in range(len(architecture) - 1):
            self.weights[i] = np.random.rand(architecture[i], architecture[i+1]) \
                * (2 * INITIAL_WEIGHT_RANGE) - INITIAL_WEIGHT_RANGE

    def train(epochs, minibatch_size, learning_rate):
        for _ in range(epochs):
            data_batch = sample_training_data(self.training_data, minibatch_size)
            # find gradient
            # update weights

    def sample_training_data(data, size):
        pass

DATA_FOLDER = '../steering/'
DATA_INFO_FILE = 'data.txt'
NETWORK_ARCHITECTURE = [1024, 128, 64, 1]
TRAINING_EPOCHS = 100
MINIBATCH_SIZE = 10
LEARNING_RATE = 0.01
TRAINING_RATIO = 0.8

def read_data():
    with open(DATA_FOLDER + DATA_INFO_FILE, 'r') as f:
        content = f.readlines()
    data_info = []
    for line in content:
        [img, deg] = line.split()
        image = cv2.imread(DATA_FOLDER + img[2:])
        data_info.append((image, deg))
    return data_info

if __name__ == "__main__":
    data = read_data()
    len_data = len(data)
    split_point = math.floor(len_data * TRAINING_RATIO)
    training_data = data[:split_point]
    testing_data = data[split_point+1:]
    print(len_data)
    print(len(training_data))
    print(len(testing_data))
    nn = Neural_Network(NETWORK_ARCHITECTURE, training_data, testing_data)
    nn.train(TRAINING_EPOCHS, MINIBATCH_SIZE, LEARNING_RATE)
