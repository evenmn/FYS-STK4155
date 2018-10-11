import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ising_data import *

class Reg():

    def __init__(self, X, y):
        '''
        Arguments:
        ----------
        
        X:  2D numpy array.
            Design matrix.
            
        y:  numpy array.
            Targets.'''
                
        self.X = X
        self.y = y


    def ols(self):
        '''Ordinary Least Square (OLS)'''
        
        X = self.X
        y = self.y

        # Calculating beta-vector
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
        return beta.flatten()



    def reg_q(self, q, λ=1e-5, η=1e-4, niter=1e5):
        '''Regression with penalty
        
        Arguments:
        ----------
        
        q:      Integer.
                Exponent in last term.
                
        λ:      Float.
                Penalty.
        
        η:      Float.
                Learning rate.
                
        niter:  Integer.
                Number of iterations in gradient descent.'''
                
        X = self.X
        y = self.y

        # Minimization using gradient descent
        beta = np.random.randn(X.shape[1], 1)
        beta = beta[:,0]

        for iter in tqdm(range(int(niter))):
            e = y - X.dot(beta)                    # Absolute error
            beta += η*(2*X.T.dot(e) - np.sign(beta)*q*λ*np.power(abs(beta), q-1))
        
        return beta.flatten()
        
        
        
    def ridge(self, λ=1e-5):
        '''Ridge regression'''
        
        X = self.X
        y = self.y

        # Calculating beta-vector
        beta = np.linalg.inv(X.T.dot(X) + λ*np.eye(X.shape[1])).dot(X.T).dot(y)
        return beta.flatten()
        
        
        
    def lasso(self, λ=1e-5, η=1e-6, niter=1e4):
        '''Lasso regression'''
        return Reg.reg_q(self, 1, λ, η, niter)
    
    
    

if __name__ == '__main__':


    # --- Simple running example ---
    # define Ising model params
    L = 4         # system size
    N = 1000000   # Number of states

    # create random Ising states
    states = np.random.choice([-1, 1], size=(N, L))
    X = np.multiply(states[:,1:], states[:,:-1])

    # calculate Ising energies
    E = ising_energies(states, L)

    model = Reg(X, E)
    J = model.ols()

    print(J)
