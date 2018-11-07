#!/usr/bin/python

import numpy as np
from numpy.random import random as rand
from activation_function import *
from transformation import f, x
from sys import exit
from tqdm import tqdm


def linear(X, t, T, eta = 0.001, minimization='GD', f=none):
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
    
    Returning
    ---------
    W  {Array}   : Weights
                   Size [I, O]
    '''
    
    I = len(X[0])
    M = len(X)
    try:
        O = len(t[0])
    except:
        O = 1
    
    if len(t) != M:
        print("Input and output arrays do not have the same length, rejecting")
        sys.exit()
    
    W = (2*np.random.random([I+1, O]) - 1)*0.001
    X = np.c_[X, np.ones(M)]
    
    if minimization == 'GD':
        for iter in tqdm(range(T)):
            for i in range(M):
                net = np.dot(X[i], W)
                out = f(net)

                deltao = (out - t[i]) * f(out, der=True)
                W = W - eta * np.outer(np.transpose(X[i]), deltao)
                
    elif minimization == 'SGD':
        m = 2000

        for epoch in tqdm(range(T)):
            for i in range(m):
                random_index = np.random.randint(m)
                Xi = X[random_index:random_index+1]
                ti = t[random_index:random_index+1]
                
                net = np.dot(Xi, W)
                out = f(net)

                deltao = -(ti - out) * f(out, der=True)
                dW = np.outer(np.transpose(Xi), deltao)

                W -= eta * np.outer(np.transpose(Xi), deltao)
    return W


def nonlinear(X, t, T, H, eta = 0.001, minimization='GD', f1=sigmoid, f2=none):
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
    H {Int}      : Number of hidden nodes.
                   Needs to be given by the user.
    eta {float}  : Learning rate.
                   Usually in the interval [0, 0.5].
    
    Returning
    ---------
    W1 {Array}   : Weights 1
                   Size [I, H] where H is the number of hidden nodes.     
    W2 {Array}   : Weights 2
                   Size [H, O] where H is the number of hidden nodes.
    '''
    
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
    W1 = (2*np.random.random([I, H]) - 1)*0.00001
    W2 = (2*np.random.random([H, O]) - 1)*0.00001

    b1 = np.random.random(H)*0.00001
    b2 = np.random.random(O)*0.00001

    # Training
    if minimization == 'GD':
        for iter in tqdm(range(T)):
            for i in range(M):
            
                # FORWARD PROPAGATION
                net_h = np.dot(X[i], W1) + b1
                out_h = f1(net_h)

                net_o = np.dot(out_h, W2) + b2
                out_o = f2(net_o)
                
                # BACKWARD PROPAGATION
                # Last weights
                deltao = (out_o - t[i]) * f1(out_o, der=True)
                
                print(deltao)
                
                # First weights            
                deltah = f2(out_h, der=True) * np.dot(deltao, np.transpose(W2))
                
                # Update weights
                b1 = b1 - eta * deltah
                b2 = b2 - eta * deltao
                W1 = W1 - eta * np.outer(X[i], deltah)
                W2 = W2 - eta * np.outer(out_h, deltao)
            
    elif minimization == 'SGD':
        m = 1000

        for epoch in tqdm(range(T)):
            error = 0
            for i in range(m):
                random_index = np.random.randint(m)
                Xi = X[random_index:random_index+1]
                ti = t[random_index:random_index+1]
                
                # FORWARD PROPAGATION
                net_h = np.dot(Xi, W1) + b1
                out_h = f1(net_h)

                net_o = np.dot(out_h, W2) + b2
                out_o = f2(net_o)
                
                # BACKWARD PROPAGATION
                # Last weights
                deltao = -(ti - out_o) * f1(out_o, der=True)
                
                # First weights            
                deltah = f2(out_h, der=True) * np.dot(deltao, np.transpose(W2))
                
                # Update weights
                b1 = b1 - eta * deltah
                b2 = b2 - eta * deltao
                W1 = W1 - eta * np.outer(Xi, deltah)
                W2 = W2 - eta * np.outer(out_h, deltao)
                
                error += abs(ti - out_o)
            print(error)
            
    return W1, W2, b1, b2


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
   
   
    
def recall_linear(X, W):
    X = np.c_[X, np.ones(len(X))]
    Out = np.empty(len(X))
    for i in range(len(X)):
        net = np.dot(X[i], W)
        out = net #sigmoid(net)
        Out[i] = out
        
    return Out
        
        
def recall_nonlinear(X, W1, W2, b1, b2):
    Out = np.empty(len(X))
    for i in range(len(X)):
        net_h = np.dot(X[i], W1) + b1
        out_h = ReLU(net_h)

        net_o = np.dot(out_h, W2) + b2
        out_o = net_o #sigmoid(net_o)
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
