#!/usr/bin/python

import numpy as np
from convert_pkl import ignore_tc
from logistic_CE import *
from error_tools import Accuracy
import neural_network as nn

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

'''
X = np.array([[0,0], [0,1], [1,0], [1,1]])
t = np.array([0,1,1,1])

W = nn.linear(X, t, 1000, eta=1, minimization='GD', trans=False)
y = nn.recall_linear(X, W, trans=False)

print(y)
stop
'''

n = 120000                 # Number of training sets

data = ignore_tc()
data[np.where(data==0)]=-1

X_train = data[:n,:-1]
t_train = data[:n,-1]

X_test = data[:n,:-1]
t_test = data[:n,-1]

T = 1000

W = nn.linear(X_train, t_train, T, eta=1, minimization='GD', trans=False)
y_test = nn.recall_linear(X_test, W, trans=False)

print('Accuracy: ', Accuracy(y_test, t_test))
