#!/usr/bin/python

import numpy as np
from convert_pkl import ignore_tc
from error_tools import Accuracy
from activation_function import *
import neural_network as nn


# Some constants
n = 120000                 # Number of training sets
T = 1                      # Number of iterations
eta = 0.0001               # Learning rate
method = 1                 # 0 is logistic regression
                           # 1 is neural networks

# Generate training set and divide into training and test
data = ignore_tc()
data[np.where(data==0)]=-1

X_train = data[:n,:-1]
t_train = data[:n,-1]

X_test = data[:n,:-1]
t_test = data[:n,-1]

if method == 0:
    # Logistic regression
    obj = nn.NeuralNetwork(X_train, t_train, T, eta=eta, f1=logistic)  # Define object 

    obj.solver()                                      # Obtain optimal weights
    y_train = obj.recall(X_train)                     # Recall training energy
    y_test = obj.recall(X_test)                       # Recall test energy

    print('Accuracy train: ', Accuracy(y_train, t_train))
    print('Accuracy test: ', Accuracy(y_test, t_test))

elif method == 1:
    # Neural network
    h = 100
    obj = nn.NeuralNetwork(X_train, t_train, T, h, eta=eta, f1=logistic, f2=Leaky_ReLU)

    obj.solver()                                      # Obtain optimal weights
    y_train = obj.recall(X_train)                     # Recall training energy
    y_test = obj.recall(X_test)                       # Recall test energy

    print('Accuracy train: ', Accuracy(y_train, t_train))
    print('Accuracy test: ', Accuracy(y_test, t_test))
