from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from regression import reg, f

def reg_q(x, y, p, q, λ=1, η=0.0001, niter=10000):
    '''Regression, finding coefficients beta
    
    Arguments:
    ----------
    
    x:      Numpy array.
            X-component of all points.
            
    y:      Numpy array.
            Y-component of all points.
            
    p:      Integer.
            Order of fitting polynomial.
            
    q:      Float.
            Degree of additional element.
            Lasso: q=1
            Ridge: q=2
            SLR:   q=-∞.
            
    eta:    Float.
            Learning rate.
            
    niter:  Integer.
            Number of iterations in gradient descent.'''
    
    n = len(x)
    
    # Creating X-matrix
    xb = np.c_[np.ones((n,1))]
    for i in range(1,p+1):
        xb = np.c_[xb, x**i]

    # Minimization using gradient descent
    beta = np.random.randn(p+1,1)
    
    for iter in range(niter):
        y_hat = xb.dot(beta)
        e = y - y_hat           # Absolute error
        beta += η*(xb.T.dot(e) + q*λ*beta**(q-1))
    
    return beta.flatten()[::-1]



if __name__ == '__main__':
    x_vals = np.linspace(0,2,100)
    x = 2*np.random.rand(100,1)
    y = 4+3*x+np.random.randn(100,1)

    plt.plot(x, y, '.')
    plt.plot(x_vals, f(x_vals, reg_q(x, y, 2, 1)), label='Lasso, $q=1$')
    plt.plot(x_vals, f(x_vals, reg_q(x, y, 2, 2)), label='Ridge, $q=2$')
    plt.plot(x_vals, f(x_vals, reg(x, y, 2)), label='Standard, $q=-\infty$')
    plt.legend()
    plt.grid()
    plt.show()
