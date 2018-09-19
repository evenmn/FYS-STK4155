import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm


class Reg_1D():

    def __init__(self, x, y, p):
        '''
        Arguments:
        ----------
        
        x:      Numpy array.
                X-component of all points.
                
        y:      Numpy array.
                Y-component of all points.
                
        n:      Integer.
                Order of fitting polynomial.'''
    
        self.x = x
        self.y = y
        self.p = p


    def ols(self):
        '''Ordinary Least Square (OLS) regression'''
        
        x = self.x
        y = self.y
        p = self.p
        
        xb = np.c_[np.ones((len(x),1))]
        for i in range(1,p+1):
            xb = np.c_[xb, x**i]

        beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
        return beta.flatten()[::-1]


    def reg_q(self, q, λ=0.1, η=0.0001, niter=100000):
        '''Generalized form of Ridge and Lasso regression
        
        Arguments:
        ----------
                
        q:      Float.
                Degree of additional element.
                Lasso: q=1
                Ridge: q=2
                SLR:   q=-∞.
                
        eta:    Float.
                Learning rate.
                
        niter:  Integer.
                Number of iterations in gradient descent.'''
        
        x = self.x
        y = self.y
        p = self.p
        
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


    def ridge(self, λ=0.1, η=0.0001, niter=100000):
        '''Ridge regression'''
        return Reg_1D.reg_q(self, 2, λ, η, niter)
        
        
    def lasso(self, λ=0.1, η=0.0001, niter=100000):
        '''Lasso regression'''
        return Reg_1D.reg_q(self, 1, λ, η, niter)
        

def f(x, pol):
    '''Polyval'''
    return np.polyval(pol, x)
    
    
if __name__ == '__main__':

    # --- Simple running example ---
    x          = 2*np.random.rand(100,1)
    y          = 4+3*x+np.random.randn(100,1)
    
    order2     = Reg_1D(x, y, 2)
    beta_ols   = order2.ols()
    beta_ridge = order2.ridge()
    beta_lasso = order2.lasso()

    
    #print(beta_ols)
    #print(beta_ridge)
    #print(beta_lasso)

    x_vals = np.linspace(0,2,100)
    
    plt.plot(x, y, '.')
    plt.plot(x_vals, f(x_vals, beta_ols), label='OLS')
    plt.plot(x_vals, f(x_vals, beta_ridge), label='ridge')
    plt.plot(x_vals, f(x_vals, beta_lasso), label='lasso')
    plt.legend(loc='best')
    plt.show()
