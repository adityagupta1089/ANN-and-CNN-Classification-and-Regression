import cv2
import math
import numpy as np
import random
from tqdm import *
import tensorflow as tf


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    # data lists
    X = []
    Y = []
    # training data lists
    X_train = []
    Y_train = []
    # testing data lists
    X_test = []
    Y_test = []

    # create network
    architecture    = [1024, 512, 64, 1]
    split_ratio     = 0.8
    epochs          = 5000
    learning_rate   = 0.01
    minibatch_size  = 64

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
        X.append(img.flatten().T / 255)
        Y.append(float(deg[:-2]))
    
    split        = math.floor(split_ratio * len(X))
    X_train = X[0:split]
    Y_train = Y[0:split]
    X_test  = X[split + 1:]
    Y_test  = Y[split + 1:]

    x = tf.placeholder("float")
    y = tf.placeholder("float")
    
    w = tf.Variable(np.random.randn(), name="weight")
    b = tf.Variable(np.random.randn(), name="bias")
    
    pred = tf.add(tf.multiply(x, w), b)
    
    cost = tf.reduce_sum(tf.pow(pred-y,2))/(2*split)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
    
        sess.run(init)
        
        for epoch in range(epochs):
            for (_x, _y) in zip(X_train, Y_train):
                sess.run(optimizer, feed_dict={x: _x, y: _y})
                
            if (epoch+1) % 1 == 0:
                c = sess.run(cost, feed_dict={x: _x, y: _y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(w), "b=", sess.run(b))
                
        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: _x, Y: _y})
        print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')
