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
        X {Array}:     Inputs
                       Size [M, I] where M is the number of training samples.
                       Contains a set of training samples.
                       NB: Needs to be a numpy array           
        t {Array}:     Targets
                       Size [M, O] where M is the number of training samples.
                       Contains a set of targets which correspond to the
                       input samples.
                       NB: Needs to be a numpy array  
        T {Int}:       Number of training cycles.
                       Needs to be given by the user.
        h {Int(s)}:    Number of hidden nodes.
                       List of ints indicate multiple hidden layers.
        eta {float}:   Learning rate.
                       Usually in the interval [0, 0.5].
        f1 {function}: Activation function on output.
                       Pure linear as default. Supports 'ReLU', 'ELU',
                       'Leaky_ReLU', 'logistic', 'tanh' and 'none'.
        f2 {function}: Activation function on hidden layers.
                       Pure linear as default. Supports 'ReLU', 'ELU',
                       'Leaky_ReLU', 'logistic', 'tanh' and 'none'.
        opt {function}:  Minimization function.
                         Gradient descent 'GD' as default, supports 
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
            self.O = len(self.t[0])
        except:
            self.O = 1
            
        # Number of hidden layers
        if isinstance(self.h, list):
            self.H = len(self.h)
            self.h.append(self.O)
        elif self.h == 0:
            self.H = 0
            self.h = [self.O]
        elif isinstance(self.h, int):
            self.H = 1
            self.h = [self.h, self.O]
        else:
            raise TypeError('h needs to be a list or int')
            
        # Initialize weights, including bias weights
        self.W=[]
        self.W.append((2*np.random.random((self.I+1, self.h[0])) - 1)*np.sqrt(1/len(self.X[0,:])))
        for i in range(self.H):
            self.W.append((2*np.random.random((self.h[i]+1, self.h[i+1])) - 1)*np.sqrt(1/self.h[i]))

        
        
    def feed_forward(X, W, f1=none, f2=none):
        '''Feed forward'''
        out = np.insert(X, 0, 1)
        Out = [out]
        for i in range(len(W)-1):
            net = np.dot(out, W[i])              # Net output to layer i
            out = f2(net)                        # Output to layer i
            out = np.insert(out, 0, 1)           # Add bias
            Out.append(out)
        net = np.dot(out, W[-1])
        Out.append(f1(net))
        return Out                              # Out contains all node values
        
        
    def backward_propagation(X, t, W, Out, f1, f2):
        '''Backward propagation'''
        deltao = (Out[-1] - t) * f1(Out[-1], der=True)
        deltah = [deltao]
        
        H = len(Out) - 2
        for i in range(H):
            delta = W[H-i].dot(np.transpose(deltah[-1])) * f2(Out[H-i], der=True)
            deltah.append(delta[:-1])
        return deltah[::-1]


    def solver(self):
        '''Linear'''
        self.initialize()
        
        if self.opt == GD:
            for iter in tqdm(range(self.T)):
                for i in range(self.M):
                    Out = NeuralNetwork.feed_forward(self.X[i], self.W, f1=self.f1, f2=self.f2)
                    deltah = NeuralNetwork.backward_propagation(self.X[i], self.t[i], self.W, Out, self.f1, self.f2)
                    for j in range(self.H + 1):
                        gradient = np.outer(np.transpose(Out[j]), deltah[j][0:])
                        self.W[j] -= self.eta * gradient
                    
        elif self.opt == SGD:
            m = 2000

            for epoch in tqdm(range(self.T)):
                for i in range(m):
                    random_index = np.random.randint(m)
                    Xi = self.X[random_index:random_index+1]
                    ti = self.t[random_index:random_index+1]
                    
                    Out = NeuralNetwork.feed_forward(self.Xi, self.W, self.f1, self.f2)
                    deltah = NeuralNetwork.backward_propagation(self.Xi, self.ti, self.W, Out, self.f1, self.f2)
                    gradient = np.outer(np.transpose(Out[0], deltah))
                    self.W -= self.eta * gradient
        return self.W


    
       
       
        
def recall(X, W, f1=none, f2=none):
    '''Recall'''
    Out = np.empty(len(X))
    for i in range(len(X)):
        Out[i] = NeuralNetwork.feed_forward(X[i], W, f1, f2)[-1]
    return Out
