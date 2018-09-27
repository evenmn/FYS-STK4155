import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class Reg_scikit():
    def __init__(self, x, y, z, Px, Py):
        '''
        Arguments:
        ----------
        
        x:      Numpy array.
                X-component of all points.
                
        y:      Numpy array.
                Y-component of all points.
                
        z:      Numpy array.
                Z-component of all points.
                
        Px:     Integer.
                Order of fitting polynomial in x-direction.
                
        Py:     Integer.
                Order of fitting polynomial in y-direction.'''
                
        self.x = x
        self.y = y
        self.z = z
        self.Px = Px+1
        self.Py = Py+1


    def set_up_X(self):
        '''Set up the design matrix'''
        
        x = self.x
        y = self.y
        Px = self.Px
        Py = self.Py

        # Setting up x-matrix
        N = len(x)
        X = np.zeros([N, Px*Py])
        for i in range(N):
            for j in range(Px):
                for k in range(Py):
                    X[i,Py*j+k] = x[i]**j*y[i]**k
                    
        return X
        
        
    def ols(self):
        '''Test of ordinary Least Square (OLS)'''
        
        z = self.z
        Px = self.Px
        Py = self.Py
        
        X = Reg_scikit.set_up_X(self)
        
        reg = LinearRegression()
        reg.fit (X, z) 
        #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
        
        return np.reshape(reg.coef_, (Px,Py))
        
        
    def ridge(self, λ=0.01):
        '''Ridge regression'''
        
        z = self.z
        Px = self.Px
        Py = self.Py

        X = Reg_scikit.set_up_X(self)
        
        reg = Ridge(alpha=λ)
        reg.fit (X, z) 
        
        return np.reshape(reg.coef_, (Px,Py))
        
        
    def lasso(self, λ=0.01):
        '''Ridge regression'''
        
        z = self.z
        Px = self.Px
        Py = self.Py

        X = Reg_scikit.set_up_X(self)
        
        reg = Lasso(alpha=λ)
        reg.fit (X, z) 
        
        return np.reshape(reg.coef_, (Px,Py))
