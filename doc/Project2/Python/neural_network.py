#!/usr/bin/python

import numpy as np
from numpy.random import random as rand
from activation_function import *
from optimization import *
from sys import exit
from tqdm import tqdm

class NeuralNetwork():
    def __init__(self, X, t, T, h=0, eta=0.001, f1=none, f2=none, opt=GD):
        '''
        Arguments
        ---------
        X {Array}:       Inputs
                         Size [M, I] where M is the number of training samples.
                         Contains a set of training samples.
                         NB: Needs to be a numpy array           
        t {Array}:       Targets
                         Size [M, O] where M is the number of training samples.
                         Contains a set of targets which correspond to the
                         input samples.
                         NB: Needs to be a numpy array  
        T {Int}:         Number of training cycles.
                         Needs to be given by the user.
        h {Int(s)}:      Number of hidden nodes.
                         List of ints indicate multiple hidden layers.
        eta {float}:     Learning rate.
                         Usually in the interval [0, 0.5].
        f1 {function}:   Activation function on output.
                         Pure linear by default. Supports 'ReLU', 'ELU',
                         'Leaky_ReLU', 'logistic', 'tanh' and 'none'.
        f2 {function}:   Activation function on hidden layers.
                         Pure linear by default. Supports 'ReLU', 'ELU',
                         'Leaky_ReLU', 'logistic', 'tanh' and 'none'.
        opt {function}:  Minimization function.
                         Gradient descent 'GD' by default, supports 
                         Stochastic Gradient Descent ('SGD') as well.
        '''
        
        self.X = X
        self.t = t
        self.T = T
        self.h = h
        self.eta = eta
        self.f1 = f1
        self.f2 = f2
        self.opt = opt
        
        
    def initialize(self):
        '''Set up constants and weights'''
        
        self.I = len(self.X[0])         # Number of input nodes
        self.M = len(self.X)            # Number of data sets
        
        # Check if number of training sets and targets are the same
        if len(self.t) != self.M:
            print("Input and output arrays do not have the same length, rejecting")
            sys.exit()
        
        # Number of output nodes
        try:
            self.O = len(self.t[0])     # If multiple outputs
        except:
            self.O = 1                  # If only one output
            
        # Number of hidden layers
        if isinstance(self.h, list):    # If multiple hidden layers
            self.H = len(self.h)
            self.h.append(self.O)
        elif self.h == 0:               # If no hidden layers
            self.H = 0
            self.h = [self.O]
        elif isinstance(self.h, int):   # If one hidden layer
            self.H = 1
            self.h = [self.h, self.O]
        else:
            raise TypeError('h needs to be a list or int')
            
        # Initialize weights, including bias weights
        self.W=[]
        self.W.append((2*rand((self.I+1, self.h[0])) - 1)*np.sqrt(1/len(self.X[0,:])))
        for i in range(self.H):
            self.W.append((2*rand((self.h[i]+1, self.h[i+1])) - 1)*np.sqrt(1/self.h[i]))

        
        
    def feed_forward(self, X):
        '''Feed forward'''
        self.out = [np.insert(X, 0, 1)]               # Add bias
        for i in range(self.H):                       # Loop over hidden nodes
            net = np.dot(self.out[-1], self.W[i])     # Net output to layer i
            Out = self.f2(net)                        # Output to layer i
            self.out.append(np.insert(Out, 0, 1))     # Add bias
        net = np.dot(self.out[-1], self.W[-1])        # Final output
        self.out.append(self.f1(net))                 # Final output with f1
        
        
        
    def backward_propagation(self, t):
        '''Backward propagation'''
        deltah = [(self.out[-1] - t) * self.f1(self.out[-1], der=True)]
        for j in range(self.H):
            delta = self.W[self.H-j].dot((deltah[-1]).T) * self.f2(self.out[self.H-j], der=True)
            deltah.append(delta[:-1])
        self.deltah=deltah[::-1]


    def solver(self):
        '''Linear'''
        self.initialize()
        if self.opt == GD:
            for iter in tqdm(range(self.T)):
                for i in range(self.M):
                    self.feed_forward(self.X[i])
                    self.backward_propagation(self.t[i])
                    for j in range(self.H + 1):
                        gradient = np.outer((self.out[j]).T, self.deltah[j][0:])
                        self.W[j] -= self.eta * gradient
                    
        elif self.opt == SGD:
            m = 2000

            for epoch in tqdm(range(self.T)):
                for i in range(m):
                    random_index = np.random.randint(m)
                    Xi = self.X[random_index:random_index+1]
                    ti = self.t[random_index:random_index+1]
                    
                    self.feed_forward(self.Xi)
                    self.backward_propagation(i)
                    for j in range(self.H + 1):
                        gradient = np.outer((self.out[j]).T, self.deltah[j][0:])
                        self.W[j] -= self.eta * gradient
        return self.W
        
        
    def recall(self, X):
        '''Recall'''
        Out = np.empty(len(X))
        for i in range(len(X)):
            self.feed_forward(X[i])
            Out[i] = self.out[-1]
        return Out
