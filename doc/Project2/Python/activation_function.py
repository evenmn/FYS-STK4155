#!/usr/bin/python

import numpy as np
from numba import njit

def sigmoid(x):
    '''Maps the argument x in the interval [0, 1]'''
    return (1 + np.exp(-x))**(-1)
    
def sig_der(x):
    '''The derivative of f(x) = 1/(1 + exp(-x))'''
    return x*(1 - x)

def tanh(x):
    '''Maps the argument x in the interval [-1, 1]'''
    return np.tanh(x)
    
def tanh_der(x):
    '''The derivative of f(x) = tanh(x)'''
    return (np.cosh)**(-2)   
    
def ReLU(x):
    '''Standard ReLU'''
    if x>0:
        return x
    else:
        return 0
        
def Leaky_ReLU(x):
    '''Leaky ReLU'''
    if x>0:
        return x
    else:
        return 0.1*x
        
def ELU(x):
    '''ELU'''
    if x>0:
        return x
    else:
        return np.exp(x)-1
        
def stepwise(x):
    '''Stepwise function'''
    if x>0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    x = np.linspace(-8, 8, 1000)  
    label_size = {'size':'16'}

    sns.set()
    
    plt.axhline( 0, linestyle = '--', color = 'r', linewidth=0.5)
    plt.axhline( 1, linestyle = '--', color = 'r', linewidth=0.5)
    plt.plot(x, sigmoid(x), linewidth = 2)

    #plt.title(r"Sigmoid function $sigmoid(x)$",**label_size)
    plt.xlabel("$x$",**label_size)
    plt.ylabel("$f(x)$",**label_size)
    plt.text(-8, 0.5, r'$f(x)=(1 + \exp(-x))^{-1}$',**{'size':'14'})
    plt.savefig("../plots/sigmoid.png")
    plt.show()

    plt.axhline(-1, linestyle = '--', color = 'r', linewidth=0.5)
    plt.axhline( 1, linestyle = '--', color = 'r', linewidth=0.5)
    plt.plot(x, tanh(x), linewidth = 2)

    #plt.title(r"Sigmoid function $f(x)=tanh(x)$",**label_size)
    plt.xlabel("$x$",**label_size)
    plt.ylabel("$f(x)$",**label_size)
    plt.text(-8, 0, r'$f(x)=tanh(x)$',**{'size':'14'})
    plt.savefig("../plots/tanh.png")
    plt.show()
    
    ReLU_array = np.empty(len(x))
    for i in range(len(x)):
        ReLU_array[i] = ReLU(x[i])
        
    plt.plot(x, ReLU_array, linewidth = 2)
    #plt.title(r"Sigmoid function $f(x)=ReLU(x)$",**label_size)
    plt.xlabel("$x$",**label_size)
    plt.ylabel("$f(x)$",**label_size)
    plt.text(-8, 4, r'f(x)=max(0,x)',**{'size':'14'})
    plt.savefig("../plots/ReLU.png")
    plt.show()
    
    LeakyReLU_array = np.empty(len(x))
    for i in range(len(x)):
        LeakyReLU_array[i] = Leaky_ReLU(x[i])
    
    plt.plot(x, LeakyReLU_array, linewidth = 2)
    #plt.title(r"Sigmoid function $f(x)=Leaky\_ReLU(x)$",**label_size)
    plt.xlabel("$x$",**label_size)
    plt.ylabel("$f(x)$",**label_size)
    plt.text(-8, 4, r'f(x)=0.1x if x<0, x else',**{'size':'14'})
    plt.savefig("../plots/LeakyReLU.png")
    plt.show()
    
    ELU_array = np.empty(len(x))
    for i in range(len(x)):
        ELU_array[i] = ELU(x[i])
    
    plt.plot(x, ELU_array, linewidth = 2)
    #plt.title(r"Sigmoid function $f(x)=ELU(x)$",**label_size)
    plt.xlabel("$x$",**label_size)
    plt.ylabel("$f(x)$",**label_size)
    plt.text(-8, 4, r'f(x)=exp(x)-1 if x<0, x else',**{'size':'14'})
    plt.savefig("../plots/ELU.png")
    plt.show()

