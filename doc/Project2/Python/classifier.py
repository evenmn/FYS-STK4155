#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Ising_2D import ignore_tc, tc
from error_tools import Accuracy
from activation import *
from optimization import *
import neural_network as nn
sns.set()

# Some constants
n = 100000                 # Number of training sets
T = 1                      # Number of iterations
eta = 0.0001               # Learning rate
method = 0                 # 0 is logistic regression
                           # 1 is neural networks

# Generate training set and divide into training and test
data = ignore_tc()
data_critical = tc()
data[np.where(data==0)] = -1
data_critical[np.where(data==0)] = -1

X_train = data[:n,:-1]
t_train = data[:n,-1]

X_test = data[n:,:-1]
t_test = data[n:,-1]

X_critical = data_critical[:,:-1]
t_critical = data_critical[:,-1]


if method == 0:
    lambdas = np.logspace(-6,3,10)
    etas = np.logspace(-6,3,10)
    train = np.empty([10,10])
    test = np.empty([10,10])
    critical = np.empty([10,10])
    
    i = 0
    for eta in etas:
        j = 0
        for lamb in lambdas:
            # Logistic regression
            obj = nn.NeuralNetwork(X_train, t_train, T, eta=eta, lamb=lamb, f1=logistic)  # Define object 

            obj.solver()                                      # Obtain optimal weights
            y_train = obj.recall(X_train)                     # Recall training phase
            y_test = obj.recall(X_test)                       # Recall test phase
            y_critical = obj.recall(X_critical)               # Recall critical phase
            
            train[i,j] = Accuracy(y_train, t_train)
            test[i,j] = Accuracy(y_test, t_test)
            critical[i,j] = Accuracy(y_critical, t_critical)
            
            print('\n', eta, lamb)
            print('Accuracy train: ', train[i,j])
            print('Accuracy test: ', test[i,j])
            print('Accuracy critical: ', critical[i,j])
            
            j += 1
        i += 1
            
    # Save matrices
    np.savetxt('../data/train.dat', train)
    np.savetxt('../data/test.dat', test)
    np.savetxt('../data/critical.dat', critical)
            
    # Heatmap
    sets = [train, test, critical]
    i = 0
    for set_ in sets:
        ax = sns.heatmap(set_, annot=True, fmt='.2f')

        #ax.set_xticks(etas)
        #ax.set_yticks(lambdas)
        
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("{}".format(i))
        plt.show()
            


elif method == 1:
    # Neural network
    h = 10
    obj = nn.NeuralNetwork(X_train, t_train, T, h, eta=eta, f1=logistic, f2=Leaky_ReLU, opt=GD)

    obj.solver()                                      # Obtain optimal weights
    y_train = obj.recall(X_train)                     # Recall training phase
    y_test = obj.recall(X_test)                       # Recall test phase
    y_critical = obj.recall(X_critical)               # Recall critical phase

    print('Accuracy train: ', Accuracy(y_train, t_train))
    print('Accuracy test: ', Accuracy(y_test, t_test))
    print('Accuracy critical: ', Accuracy(y_critical, t_critical))
