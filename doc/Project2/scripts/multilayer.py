#!/usr/bin/python

import neural_network as nn
import numpy as np
from numpy.random import random as rand
from sigmoid import sigmoid1, sig_der1
from sys import exit
from tqdm import tqdm
#from numba import njit, prange

#@njit(nopython=False, parallel=True)
def multilayer(X, t, T, h, eta = 0.1):
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
        raise TypeError("h needs to be a numpy array")
        
    if np.count_nonzero(h) != len(h):
        print("h contains one or more zeros, trying to solve it linearly")
        W, b = nn.linear(X, t, T)
        return W, b
        exit()
        

    I = len(X[0])
    O = len(t)
    M = len(X)
    H = len(h)
    
    if len(t) != M:
        raise ValueError("Input and output array do not have the same length, rejecting")

    # Weights
    W = []; b = []
    W.append(2 * rand([I, h[0]]) - 1)           # Add first W-matrix
    b.append(2 * rand(h[0]) - 1)                # Add first bias vector
    
    for i in range(H-1):
        W.append(2 * rand([h[i],h[i+1]]) - 1)   # Add other W-matrices
        b.append(2 * rand(h[i+1]) - 1)          # Add other bias vectors
    W.append(2 * rand([h[-1], O]) - 1)          # Add last W-matrix
    b.append(2 * rand(O) - 1)                   # Add last bias vector
    W = np.array(W)
    b = np.array(b)

    # Training
    for iter in tqdm(range(T)):
        for i in tqdm(range(M)):
        
            # FORWARD PROPAGATION
            out = []
            out.append(X[i])
            for j in range(H+1):
                net = np.dot(out[j], W[j]) + b[j]
                out.append(sigmoid1(net))
            
            
            # BACKWARD PROPAGATION
            deltao = -(t[i] - out[-1]) * sig_der1(out[-1])
            
            deltah = []
            deltah.append(deltao)
            for j in range(H):
                delta = np.dot(W[H-j], np.transpose(deltah[-1])) * sig_der1(out[H-j])
                deltah.append(delta)
            deltah = deltah[::-1]     
            
            # Update weights
            for j in range(H + 1):
                W[j] = W[j] - eta * np.outer(out[j], deltah[j])
                b[j] = b[j] - eta * deltah[j]
            
    return W, b

#@njit(parallel=True)
def recall_multilayer(X, W, b):
    Out = np.empty(len(X))
    for i in range(len(X)):
        out = []
        out.append(X[i])
        for j in range(len(W)):
            net = np.dot(out[j], W[j]) + b[j]
            out.append(sigmoid1(net))
        Out[i] = out[-1]
    return Out
