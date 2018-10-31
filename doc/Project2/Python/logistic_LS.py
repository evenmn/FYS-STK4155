import numpy as np
from tqdm import tqdm
from sigmoid import *

# Single perceptron for linear problems
def logistic(X, t, T, eta = 0.1):
    '''
    Logistic regression based on least square cost function
    
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
    
    np.append(X, 1)                         # Add bias to X
    
    I = len(X[0])
    M = len(X)
    
    if len(t) != M:
        print("Input and output array do not have the same length, rejecting")
        sys.exit()
    
    W = 2*np.random.random(I) - 1

    for iter in tqdm(range(T)):
        for i in tqdm(range(M)):
            net = np.dot(X[i], W)
            out = sigmoid1(net)

            deltao = -(t[i] - out) * sig_der1(out)
            W -= eta * np.dot(deltao, X[i])
            
    return W
    
    
def recall_logistic(X, W):
    Out = np.empty(len(X))
    for i in range(len(X)):
        net = np.dot(X[i], W)
        out = sigmoid1(net)
        Out[i] = out
    return Out
    

X = [[0,0], [0,1], [1,0], [1,1]]
t = [0, 1, 1, 1]
T = 1000

W = logistic(X, t, T)
print(recall_logistic(X, W))
