#!/usr/bin/python

import numpy as np
from numpy.random import random as rand
from activation_function import *
from optimization import *
from sys import exit
from tqdm import tqdm

class NeuralNetwork():
    def __init__(self, X, t, T, h=0, eta=0.001, opt=GD):
        '''
        Arguments
        ---------
        X {Array}    : Inputs
                       Size [M, I] where M is the number of training samples.
                       Contains a set of training samples.
                       NB: Needs to be a numpy array           
        t {Array}    : Targets
                       Size [M, O] where M is the number of training samples.
                       Contains a set of targets which correspond to the
                       input samples.
                       NB: Needs to be a numpy array  
        T {Int}      : Number of training cycles.
                       Needs to be given by the user.
        eta {float}  : Learning rate.
                       Usually in the interval [0, 0.5].
        
        '''
        
        self.X = X
        self.t = t
        self.T = T
        self.h = h
        self.eta = eta
        self.opt = opt
        
        
    def initialize(self):
        '''Initialize'''
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
            self.H = len(h)
        elif self.h == 0:
            self.H = 0
            self.h = [0]
        elif isinstance(self.h, int):
            self.H = 1
            self.h = [self.h]
        else:
            raise TypeError('h needs to be a list or int')
        
        # Initialize weights, including bias weights
        self.W=[]
        self.W.append((2*np.random.random([self.I+1, self.O]) - 1)*np.sqrt(1/len(self.X[0,:])))
        for i in range(self.H):
            self.W.append((2*np.random.random([self.h[i]+1, self.O]) - 1)*np.sqrt(1/self.h[i]))
        
        
        
    def feed_forward(X, W, f1=none, f2=none):
        '''Feed forward'''
        n_layers = len(W)
        Out = [np.insert(X, 0, 1)]
        for i in range(n_layers-1):
            net = np.dot(Out[i], W[i])             # Net output to layer i
            out = f2(net)                  # Output to layer i
            Out.append(np.insert(out, 0, 1))       # Add bias
        net = np.dot(Out[-1], W[-1])
        Out.append(f1(net))
        return Out                              # Output from network
        
        
    def backward_propagation(X, t, out, f):
        '''Backward propagation'''
        deltao = (out - t) * f(out, der=True)
        return np.outer(np.transpose(np.insert(X, 0, 1)), deltao)


    def linear(self, f=none):
        '''Linear'''
        self.initialize()
        
        if self.opt == GD:
            for iter in tqdm(range(self.T)):
                for i in range(self.M):
                    Out = NeuralNetwork.feed_forward(self.X[i], self.W, f1=f)
                    gradient = NeuralNetwork.backward_propagation(self.X[i], self.t[i], Out[-1], f)
                    self.W -= self.eta * gradient
                    
        elif self.opt == SGD:
            m = 2000

            for epoch in tqdm(range(self.T)):
                for i in range(m):
                    random_index = np.random.randint(m)
                    Xi = self.X[random_index:random_index+1]
                    ti = self.t[random_index:random_index+1]
                    
                    Out = NeuralNetwork.feed_forward(self.Xi, [self.W], f)
                    gradient = NeuralNetwork.backward_propagation(self.Xi, self.ti, Out[-1], f)
                    self.W -= self.eta * gradient
        return self.W


    def nonlinear2(self, f1=none, f2=none):
        
        self.initialize()

        # Training
        if self.opt == GD:
            for iter in tqdm(range(self.T)):
                for i in range(self.M):
                
                    # FORWARD PROPAGATION
                    Out = NeuralNetwork.feed_forward(self.X[i], self.W, f1)
                    
                    # BACKWARD PROPAGATION
                    # Last weights
                    deltao = (Out[-1] - self.t[i]) * f1(Out[-1], der=True)
                    
                    # First weights            
                    deltah = f2(Out[-2], der=True) * np.dot(deltao, np.transpose(self.W[-1]))
                    
                    # Update weights
                    self.W[0] -= self.eta * np.outer(self.X[i], deltah[:-1])
                    self.W[1] -= self.eta * np.outer(Out[-2], deltao)
                
        '''
        elif opt == SGD:
            m = 1000

            for epoch in tqdm(range(T)):
                error = 0
                for i in range(m):
                    random_index = np.random.randint(m)
                    Xi = X[random_index:random_index+1]
                    ti = t[random_index:random_index+1]
                    
                    # FORWARD PROPAGATION
                    net_h = np.dot(Xi, W1)
                    out_h = f1(net_h)
                    np.insert(out_h, 0, 1)

                    net_o = np.dot(out_h, W2)
                    out_o = f2(net_o)
                    
                    # BACKWARD PROPAGATION
                    # Last weights
                    deltao = -(ti - out_o) * f1(out_o, der=True)
                    
                    # First weights            
                    deltah = f2(out_h, der=True) * np.dot(deltao, np.transpose(W2))
                    
                    # Update weights
                    W1 = W1 - eta * np.outer(Xi, deltah)
                    W2 = W2 - eta * np.outer(out_h, deltao)
        '''
        return self.W


def nonlinear(self, f1=none, f2=none):
        
    I = len(X[0])
    M = len(X)
    try:
        O = len(t[0])
    except:
        O = 1
    
    if len(t) != M:
        print("Input and output array do not have the same length, rejecting")
        sys.exit()

    # Weights
    W1 = (2*np.random.random([I+1, H]) - 1)*np.sqrt(1/len(X[0,:]))
    W2 = (2*np.random.random([H+1, O]) - 1)*np.sqrt(1/H)
    X = np.c_[np.ones(len(X)),X]

    # Training
    if minimization == 'GD':
        for iter in tqdm(range(T)):
            for i in range(M):
            
                # FORWARD PROPAGATION
                net_h = np.dot(X[i], W1)
                out_h = f1(net_h)
                out_h = np.insert(out_h, 0, 1)

                net_o = np.dot(out_h, W2)
                out_o = f2(net_o)
                
                # BACKWARD PROPAGATION
                # Last weights
                deltao = (out_o - t[i]) * f1(out_o, der=True)
                
                # First weights            
                deltah = f2(out_h, der=True) * np.dot(deltao, np.transpose(W2))
                #print(np.outer(X[i], deltah).shape)
                #print(W1.shape)
                
                # Update weights
                W1 -= eta * np.outer(X[i], deltah[:-1])
                W2 -= eta * np.outer(out_h, deltao)
            
    elif minimization == 'SGD':
        m = 1000

        for epoch in tqdm(range(T)):
            error = 0
            for i in range(m):
                random_index = np.random.randint(m)
                Xi = X[random_index:random_index+1]
                ti = t[random_index:random_index+1]
                
                # FORWARD PROPAGATION
                net_h = np.dot(Xi, W1)
                out_h = f1(net_h)
                np.insert(out_h, 0, 1)

                net_o = np.dot(out_h, W2)
                out_o = f2(net_o)
                
                # BACKWARD PROPAGATION
                # Last weights
                deltao = -(ti - out_o) * f1(out_o, der=True)
                
                # First weights            
                deltah = f2(out_h, der=True) * np.dot(deltao, np.transpose(W2))
                
                # Update weights
                W1 = W1 - eta * np.outer(Xi, deltah)
                W2 = W2 - eta * np.outer(out_h, deltao)
            
    return W1, W2


def multilayer(X, t, T, h, eta = 0.001):
    '''
    Arguments
    ---------
    X {Array}    : Inputs
                   Size [M, I] where M is the number of training samples.
                   Contains a set of training samples.
                   NB: Needs to be a numpy array
    t {Array}    : Targets
                   Size [M, O] where M is the number of training samples.
                   Contains a set of targets which correspond to the
                   input samples.
                   NB: Needs to be a numpy array
    T {Int}      : Number of training cycles.
                   Needs to be given by the user.
    h {Int}      : Array with the number of hidden nodes,
                   Ex: h = np.array([5,4]) means two hidden layers with 5 and 4
                   hidden nodes respectively. 
                   Needs to be given by the user.
    eta {float}  : Learning rate.
                   Usually in the interval [0, 0.5].
    
    Returning
    ---------
    W {list}     : List of all the weight matrices.
                   Will have length len(h) + 1.     
    b {list}     : List of all the bias arrays.
                   Will have length len(h) + 1.
    '''
    
    if type(h) is not np.ndarray:
        if type(h) is int:
            h = np.array([h])
        if type(h) is list:
            h = np.array(h)
        else:
            raise TypeError("h needs to be a numpy array")
    
    if np.count_nonzero(h) != len(h):
        print("h contains one or more zeros, trying to solve it linearly")
        W = linear(X, t, T, eta)
        return W
        exit()
        

    I = len(X[0])
    M = len(X)
    H = len(h)
    try:
        O = len(t[0])
    except:
        O = 1
    
    if len(t) != M:
        raise ValueError("Input and output array do not have the same length, rejecting")


    #X = np.c_[X, np.ones(M)]

    # Weights
    W = []; b = []
    W.append((2 * rand([I, h[0]]) - 1)*0.0001)           # Add first W-matrix
    b.append(2 * rand(h[0]) - 1)                # Add first bias vector
    
    for i in range(H-1):
        W.append((2 * rand([h[i],h[i+1]]) - 1)*0.0001)   # Add other W-matrices
        b.append(2 * rand(h[i+1]) - 1)          # Add other bias vectors
    W.append((2 * rand([h[-1], O]) - 1)*0.0001)          # Add last W-matrix
    b.append(2 * rand(O) - 1)                   # Add last bias vector
    W = np.array(W)
    b = np.array(b)

    # Training
    for iter in tqdm(range(T)):
        error = 0
        for i in range(M):
        
            # FORWARD PROPAGATION
            out = []
            out.append(X[i])
            for j in range(H+1):
                net = (out[j]).dot(W[j]) + b[j]
                out.append(sigmoid(net))
            
            # BACKWARD PROPAGATION
            deltao = -(sigmoid(t[i]) - out[-1]) * sig_der(out[-1])
            
            deltah = []
            deltah.append(deltao)
            for j in range(H):
                delta = W[H-j].dot(np.transpose(deltah[-1])) * sig_der(out[H-j])
                deltah.append(delta)
            deltah = deltah[::-1]     
            
            # Update weights
            for j in range(H + 1):
                W[j] -= eta * np.outer(out[j], deltah[j])
                b[j] = b[j] - eta * deltah[j]
                
            error += abs(t[i] - out[-1])
        print(error)
            
            
    return W, b
       
       
        
def recall_linear(X, W, f=none):
    
    Out = np.empty(len(X))
    for i in range(len(X)):
        Out[i] = NeuralNetwork.feed_forward(X[i], [W], f)[-1]
    return Out
        
        
def recall_nonlinear(X, W1, W2, f1=none, f2=none):
    X = np.c_[np.ones(len(X)),X]
    Out = np.empty(len(X))
    for i in range(len(X)):
        net_h = np.dot(X[i], W1)
        out_h = f1(net_h)
        out_h = np.insert(out_h, 0, 1)

        net_o = np.dot(out_h, W2)
        out_o = f2(net_o)
        Out[i] = out_o
    return Out
        

def recall_multilayer(X, W, b):
    #X = np.c_[X, np.ones(len(X))]
    Out = np.empty(len(X))
    for i in range(len(X)):
        out = []
        out.append(X[i])
        for j in range(len(W)):
            net = np.dot(out[j], W[j]) + b[j]
            out.append(sigmoid(net))
        Out[i] = out[-1]
    return Out
