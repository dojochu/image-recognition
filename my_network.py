__author__ = 'skao'

import numpy as np
from pandas.io.parsers import read_csv
import random
import sys
import os
sys.path.insert(0, '../neural-networks-and-deep-learning/src/')
import network as net


class My_Network:

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, inputs):
        activation = [inputs]
        for b,w in zip(self.biases, self.weights):
            inputs = sigmoid(np.dot(w,inputs).reshape(w.shape[0],1)+b)
            activation.append(inputs)

        return activation

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for iter in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:(k+mini_batch_size)] for k in range(0,n,mini_batch_size)]


            self.update_mini_batch(mini_batches, eta)

            if test_data:
                print "Epoch {0}: {1}/{2} ".format(iter, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} completed".format(iter)



    def backprop(self, x, y):

        n_layers = self.weights.__len__()
        m = len(x)

        gradient_weight = []
        gradient_biases = []
        for w, b in zip(self.weights, self.biases):
            gradient_weight.append(np.zeros([w.shape[0],w.shape[1]]))
            gradient_biases.append(np.zeros([b.shape[0],1]))

        for ind in range(0,m):

            activation = self.feedforward(np.array(x[ind]).reshape(len(x[ind]),1))
            delta_error = (activation[n_layers] - y[ind].reshape(activation[n_layers].shape[0],1))*sigmoid_prime(np.dot(self.weights[(n_layers-1)],activation[n_layers-1]) + self.biases[(n_layers-1)])

            for layer in xrange(0,n_layers).__reversed__():

                i = delta_error.shape[0]
                j = activation[layer].shape[0]

                gradient_biases[layer] = (gradient_biases[layer]*(ind) + delta_error)/(ind + 1)
                gradient_weight[layer] = (gradient_weight[layer]*(ind) +  np.multiply(delta_error, activation[layer].transpose()))/(ind+1)
                if layer > 0:
                    delta_error = np.dot(self.weights[layer].transpose(),delta_error)*sigmoid_prime(np.dot(self.weights[layer-1],activation[layer-1]).reshape(self.weights[layer-1].shape[0],1) + self.biases[(layer-1)])

        return (gradient_weight,gradient_biases)


    def update_mini_batch(self, mini_batch, eta):

        for data in mini_batch:
            x_data = [x for x,y in data]
            y_data = [y for x,y in data]
            delta_w, delta_b = self.backprop(x_data,y_data)

            for layer in range(1,len(self.weights)):
                self.weights[layer] = self.weights[layer] - eta*delta_w[layer]
                self.biases[layer] = self.biases[layer] - eta*delta_b[layer]

    def evaluate(self, test_data):
        results = []
        label = []
        for x, y in test_data:
            a = self.feedforward(x)
            results.append(a[len(self.weights)])
            label.append(y)
        return (label,results)

    def cost(self, y, results):
        sum = 0
        for y_d, r_d in zip(y,results):
            sum = sum + np.sum(np.power(y_d-r_d,2))
        return sum/2/len(y)

def sigmoid(value):
    return 1.0/(1.0 + np.exp(-value))

def sigmoid_prime(value):
    return np.exp(value)/np.power(1+np.exp(value),2)
