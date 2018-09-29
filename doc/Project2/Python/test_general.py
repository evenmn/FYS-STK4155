#!/usr/bin/python

import numpy as np
import sys

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

# Sigmoid function
def sigmoid(x):
    '''Maps the argument x in the interval [0, 1]'''
    return (1 + np.exp(-x))**(-1)
    
def sig_der(x):
    '''The derivative of f(x) = 1/(1 + exp(-x))'''
    return x*(1 - x)


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
            out = sigmoid(net)

            W_new = W -eta*np.outer(np.transpose(X[i]), (out - t[i]))
            W = W_new
    return W


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
    
    if isinstance(H, int):
        H = [H]
        print(H)
    
    L = len(H)
    I = len(X[0])
    O = len(t[0])
    M = len(X)
    
    if len(t) != M:
        print("Input and output array do not have the same length, rejecting")
        sys.exit()

    # Weights
    W = []
    b = []
    W1 = 2*np.random.random([I, H[0]]) - 1
    W.append(W1)
    b1 = np.random.random(H[0])
    b.append(b1)
    
    # Creating weights when L > 1
    for h in range(L - 1):
        W0 = 2*np.random.random([H[h], H[h+1]]) - 1
        b0 = np.random.random(H[h+1])
        W.append(W0)
        b.append(b0)
    
    W2 = 2*np.random.random([H[-1], O]) - 1
    W.append(W2)

    b2 = np.random.random(O)
    b.append(b2)
    
    W = np.array(W)
    b = np.array(b)

    # Training
    for iter in range(T):
        for i in range(M):
        
            # FORWARD PROPAGATION
            net_h = np.dot(X[i], W1) + b1
            out_h = sigmoid(net_h)
            
            out = np.empty(L - 1)
            for h in range(L - 1):
                net_h = np.dot(out_h, W[h+1])
                out_h = sigmoid(net_h)
                out[h] = out_h

            net_o = np.dot(out_h, W2) + b2
            out_o = sigmoid(net_o)
            #print("Outputs: ", out_o)

            # Total error
            E_TOT = sum(0.5*((t[i] - out_o)**2))
            #print("Error:   ", E_TOT, "\n")
            
            # BACKWARD PROPAGATION
            # Last weights
            deltao = -(t[i] - out_o)*sig_der(out_o)
            deltaE = np.outer(deltao, out_h)

            W2_new = W2 - eta * np.transpose(deltaE)
            
            # Middle weights
            for h in range(L - 1):
                E = np.empty([H[L-h-2], H[L-h-1]])
                for j in range(H[L-h-2]):
                    for k in range(H[L-h-1]):
                        E[j,k] = sum(deltao*W[L-h-1][k,:]*sig_der(out[L-h-1])*X[i,j])
                W1_new = W1 - eta * E
            
            # First weights
            E = np.empty([I, H[0]])
            for j in range(I):
                for k in range(H[0]):
                    E[j,k] = sum(deltao*W2[k,:]*sig_der(out_h[k])*X[i,j])
            W1_new = W1 - eta * E

            # Update weights
            W1 = W1_new
            W2 = W2_new
            
    return W1, W2


def recall_linear(X, W):
    Out = np.empty(len(X))
    for i in range(len(X)):
        net = np.dot(X[i], W)
        out = sigmoid(net)
        Out[i] = out
    return Out
        

def recall_nonlinear(X, W1, W2):
    Out = np.empty(len(X))
    for i in range(len(X)):
        net_h = np.dot(X[i], W1)
        out_h = sigmoid(net_h)

        net_o = np.dot(out_h, W2)
        out_o = sigmoid(net_o)
        Out[i] = out_o
    return Out

X = np.array([[0,0], [1,0], [0,1], [1,1]])
t = np.array([[0], [1], [1], [0]])

W1, W2 = nonlinear(X, t, 10000, [5,5])      # Training - 10000 cycles and 5 hidden nodes
