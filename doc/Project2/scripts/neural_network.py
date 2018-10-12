#!/usr/bin/python

import sys
from sigmoid import *
from multilayer import multilayer, recall_multilayer
#from numba import njit, prange
from tqdm import tqdm


'''
This is supposed to be a class of neural networks, for different purposes.
Currently only two neural networks are included, one without hidden layer 
and one with a hidden layer. The goal is to extend this to more layers,
and make class structure.

Length input           : I
Length output          : O
Number of layers       : 2
Number of hidden nodes : H
'''


# Single perceptron for linear problems
def linear(X, t, T, eta = 0.1):
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
    O = len(t[0])
    M = len(X)
    
    if len(t) != M:
        print("Input and output array do not have the same length, rejecting")
        sys.exit()
    
    W = 2*np.random.random([I, O]) - 1
    b = 2*np.random.random(O) - 1

    for iter in range(T):
        for i in range(M):
            net = np.dot(X[i], W) + b
            out = sigmoid1(net)

            deltao = -(t[i] - out) * sig_der1(out)
            W = W - eta * np.outer(np.transpose(X[i]), deltao)
            b = b - eta * deltao
            
    return W, b


# Double perceptron for nonlinear problems

def nonlinear(X, t, T, H, eta = 0.1):
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
    O = len(t)
    M = len(X)
    
    if len(t) != M:
        print("Input and output array do not have the same length, rejecting")
        sys.exit()

    # Weights
    W1 = 2*np.random.random([I, H]) - 1
    W2 = 2*np.random.random([H, O]) - 1

    b1 = np.random.random(H)
    b2 = np.random.random(O)

    # Training
    for iter in tqdm(range(T)):
        for i in range(M):
        
            # FORWARD PROPAGATION
            net_h = np.dot(X[i], W1) + b1
            out_h = sigmoid1(net_h)

            net_o = np.dot(out_h, W2) + b2
            out_o = sigmoid1(net_o)
            
            # BACKWARD PROPAGATION
            # Last weights
            deltao = -(t[i] - out_o) * sig_der1(out_o)
            
            # First weights            
            deltah = sig_der1(out_h) * np.dot(deltao, np.transpose(W2))
            
            # Update weights
            b1 = b1 - eta * deltah
            b2 = b2 - eta * deltao
            W1 = W1 - eta * np.outer(X[i], deltah)
            W2 = W2 - eta * np.outer(out_h, deltao)
            
    return W1, W2, b1, b2


def recall_linear(X, W, b):
    Out = np.empty(len(X))
    for i in range(len(X)):
        net = np.dot(X[i], W) + b
        out = sigmoid1(net)
        Out[i] = out
    return Out
        

def recall_nonlinear(X, W1, W2, b1, b2):
    Out = np.empty(len(X))
    for i in range(len(X)):
        net_h = np.dot(X[i], W1) + b1
        out_h = sigmoid1(net_h)

        net_o = np.dot(out_h, W2) + b2
        out_o = sigmoid1(net_o)
        Out[i] = out_o
    return Out
