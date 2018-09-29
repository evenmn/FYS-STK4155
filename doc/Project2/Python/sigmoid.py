#!/usr/bin/python

import numpy as np

def sigmoid1(x):
    '''Maps the argument x in the interval [0, 1]'''
    return (1 + np.exp(-x))**(-1)
    
def sig_der1(x):
    '''The derivative of f(x) = 1/(1 + exp(-x))'''
    return x*(1 - x)

def sigmoid2(x):
    '''Maps the argument x in the interval [-1, 1]'''
    return np.tanh(x)
    
def sig_der2(x):
    '''The derivative of f(x) = tanh(x)'''
    return (np.cosh)**(-2)   

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    dirpath = os.getcwd()

    x = np.linspace(-8, 8, 1000)  

    sns.set()
    plt.axhline( 0, linestyle = '--', color = 'r', linewidth=0.5)
    plt.axhline( 1, linestyle = '--', color = 'r', linewidth=0.5)
    plt.plot(x, sigmoid1(x), linewidth = 2)

    plt.title(r"Sigmoid function $f(x)=(1 + \exp(-x))^{-1}$")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.savefig(os.path.dirname(dirpath) + "/Documents/images/sigmoid1.png")
    plt.show()

    plt.axhline(-1, linestyle = '--', color = 'r', linewidth=0.5)
    plt.axhline( 1, linestyle = '--', color = 'r', linewidth=0.5)
    plt.plot(x, sigmoid2(x), linewidth = 2)

    plt.title(r"Sigmoid function $f(x)=tanh(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.savefig(os.path.dirname(dirpath) + "/Documents/images/sigmoid2.png")
    plt.show()

