import numpy as np
from tqdm import tqdm
from sigmoid import *

def logistic(X, t, T, eta = 0.1):
    '''
    Logistic regression based on Cross-Entropy cost function
    
    Arguments
    ---------
    X {Array}    : Inputs
                   Size [M, I] where M is the number of training samples.
                   Contains a set of training samples.
                   NB: Needs to be a numpy array           
    t {Array}    : Targets
                   Size [M] where M is the number of training samples.
                   Contains a set of targets which correspond to the
                   input samples.
                   NB: Needs to be a numpy array  
    T {Int}      : Number of training cycles.
                   Needs to be given by the user.
    eta {float}  : Learning rate.
                   Usually in the interval (0, 0.5].
    
    Returning
    ---------
    W  {Array}   : Weights
                   Size [I]
    '''
    
    if len(t) != len(X):
        print("Input and output array do not have the same length, rejecting")
        sys.exit()
    
    X = np.c_[X, np.ones(len(X))[:,np.newaxis]]          # Add bias to X
    W = 2*np.random.random(len(X[0])) - 1               # Initialize W

    for iter in tqdm(range(T)):
        out = recall_logistic(X, W)                      # Forward phase
        W -= eta * (out - t).T.dot(X)                    # Backward phase     
    return W
    
    
def recall_logistic(X, W, add_bias=False):
    if add_bias:
        X = np.c_[X, np.ones(len(X))[:,np.newaxis]]      # Add bias to X
    net = X.dot(W)                                       # Netto output
    return sigmoid1(net)                                 # Activate output
    
    
if __name__ == '__main__':
    '''Simple example: OR-GATE'''
    
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    t = np.array([0, 1, 1, 1])
    T = 10000

    W = logistic(X, t, T)
    print(recall_logistic(X, W, add_bias=True))
