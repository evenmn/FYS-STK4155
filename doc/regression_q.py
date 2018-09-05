import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from regression import reg, f
from tqdm import tqdm

def reg_q(x, y, p, q, λ=0.1, η=0.0001, niter=10000):
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
    
    # Setting up X-matrix
    xb = np.c_[np.ones((len(x), 1))]
    for i in range(1, p+1):
        xb = np.c_[xb, x**i]

    # Minimization using gradient descent
    beta = np.random.randn(p+1, 1)
    for iter in tqdm(range(niter)):
        e = y - xb.dot(beta)                    # Absolute error
        beta += η*(2*xb.T.dot(e) - q*λ*np.power(abs(beta), q-1))
    
    return beta.flatten()[::-1]



if __name__ == '__main__':
    # Parameters
    npoints = 100
    degree  = 3
    x_max   = 2
    λ       = 2

    # Regression points
    x_vals = np.linspace(0, x_max, 1000)
    x      = x_max*np.random.rand(npoints,1)
    y      = 4+3*x+np.random.randn(npoints,1)
    
    # Do regression and plot
    plt.plot(x, y, '.')
    plt.plot(x_vals, f(x_vals, reg_q(x, y, degree, 1, λ=λ)), label='Lasso, $q=1$')
    plt.plot(x_vals, f(x_vals, reg_q(x, y, degree, 2, λ=λ)), label='Ridge, $q=2$')
    plt.plot(x_vals, f(x_vals, reg(x, y, degree)), label='Standard')
    plt.title('Regression with $\lambda$={}'.format(λ))
    plt.legend()
    plt.grid()
    plt.show()
