#!/usr/bin/python

import numpy as np
from convert_pkl import ignore_tc
from logistic_CE import *
from error_tools import Accuracy

'''
neural_network.py is an 1- or 2-layer artificial neural network which in principle 
can be used to solve any problem, and in practice can be used to solve a lot
of problems. In this main I will give some examples on how to use it. The input
are of dynamic size, but need to be given as 2D numpy arrays.

X - Input array
t - Target array
N - Number of training iterations
H - Number of hidden nodes
(eta - Learning rate, set to 0.1, but can be changed as last argument)
'''

n_train = 129000                 # Number of training sets
n_test = 1000                      # number of test sets

data = ignore_tc()
X_train = data[:n_train,:-1]
t_train = data[:n_train,-1]

X_test = data[n_train:n_train+n_test,:-1]
t_test = data[n_train:n_train+n_test,-1]

T = 100

W = logistic(X_train, t_train, T)
y_test = recall_logistic(X_test, W, add_bias=True)

print('Accuracy: ', Accuracy(y_test, t_test))
