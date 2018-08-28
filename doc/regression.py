from random import random, seed
import numpy as np
import matplotlib.pyplot as plt


def reg(x, y, n):
    '''Regression, finding coefficients beta
    
    Arguments:
    ----------
    
    x:      Numpy array.
            X-component of all points.
            
    y:      Numpy array.
            Y-component of all points.
            
    n:      Integer.
            Order of fitting polynomial.'''
    
    xb = np.c_[np.ones((len(x),1))]
    for i in range(1,n+1):
        xb = np.c_[xb, x**i]

    beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
    return beta.flatten()[::-1]


def f(x, pol):
    '''Polyval'''
    return np.polyval(pol, x)


if __name__ == '__main__':
    x_vals = np.linspace(0,2,100)
    x = 2*np.random.rand(100,1)
    y = 4+3*x+np.random.randn(100,1)

    plt.plot(x, y, '.')
    plt.plot(x_vals, f(x_vals, reg(x, y, 2)))
    plt.show()
