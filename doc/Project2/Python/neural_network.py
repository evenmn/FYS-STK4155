#!/usr/bin/python

import numpy as np
from numpy.random import random as rand
from activation_function import sigmoid, sig_der
from sys import exit
from tqdm import tqdm
from transformation import f, x
import time


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
    M = len(X)
    try:
        O = len(t[0])
    except:
        O = 1
    
    if len(t) != M:
        print("Input and output array do not have the same length, rejecting")
        sys.exit()
    
    W = 2*np.random.random([I, O]) - 1
    b = 2*np.random.random(O) - 1

    for iter in tqdm(range(T)):
        for i in range(M):
            net = np.dot(X[i], W) + b
            out = sigmoid(net)

            deltao = -(t[i] - out) * sig_der(out)
            W = W - eta * np.outer(np.transpose(X[i]), deltao)
            b = b - eta * deltao
            
    return W, b

    
    
def linear2(X, t, T, eta = 0.000000001):
    '''
    Arguments
    ---------
    X {Array}    : Inputs
                   Size [M, I] where M is the number of training samples.
                   Contains a set of training samples.
                   NB: Needs to be a numpy array           
    t {Array}    : TargetsS
                   Size [M, O] where M is the number of training samples.
                   Contains a set of targets which correspond to the
                   input samples.
                   NB: Needs to be a numpy array  
    T {Int}      : Number of training cycles.
                   Needs to be given by the user.
    eta {float}  : Learning rate.S
                   Usually in the interval [0, 0.5].
    
    Returning
    ---------
    W  {Array}   : Weights
                   Size [I, O]
    '''
    
    
    if len(t) != len(X):
        print("Input and output array do not have the same length, rejecting")
        sys.exit()
    
    X = np.c_[np.ones(len(X)),X]
    W = (2*np.random.random(len(X[0])) - 1)*0.01

    for iter in range(T):
        #time.sleep(0.1)
        out = X.dot(W)
        
        print('Sum: ', np.sum(np.fabs(out - t)))
        #print('Gradient: ', (np.fabs(out - t)).T.dot(X))
        
        if iter == 199999999:
            print(t)
            print(out)
            print(out-t)
        
        W -= eta * (out - t).T.dot(X) 
            
    return W



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
    
    t = f(t, -100, 100)
    
    if type(h) is not np.ndarray:
        raise TypeError("h needs to be a numpy array")
        
    if np.count_nonzero(h) != len(h):
        print("h contains one or more zeros, trying to solve it linearly")
        W, b = nn.linear(X, t, T)
        return W, b
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
                b[j] -= eta * deltah[j]
            
    return W, b


def recall_linear(X, W):
    X = np.c_[np.ones(len(X)),X]
    
    return X.dot(W)


def recall_multilayer(X, W, b):
    Out = np.empty(len(X))
    for i in range(len(X)):
        out = []
        out.append(X[i])
        for j in range(len(W)):
            net = np.dot(out[j], W[j]) + b[j]
            out.append(sigmoid(net))
        Out[i] = out[-1]
    
    Out = x(Out, -100, 100)
    return Out
