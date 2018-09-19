import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from tqdm import tqdm
from franke import FrankeFunction

# Matplotlib stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Reg_2D():

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
        self.Px = Px
        self.Py = Py


    def ols(self):
        '''Regression, finding coefficients beta'''
        
        x = self.x
        y = self.y
        z = self.z
        Px = self.Px
        Py = self.Py

        # Setting up x-matrix
        N = len(x)
        Px = Px+1; Py = Py+1
        X = np.zeros([N, Px*Py])
        for i in range(N):
            for j in range(Px):
                for k in range(Py):
                    X[i,Py*j+k] = x[i]**j*y[i]**k

        # Calculating beta-vector
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
        return np.reshape(beta.flatten(), (Px,Py))


    def reg_q(self, q, λ=0.1, η=0.0001, niter=100000):
        '''Regression, finding coefficients beta
        
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
                
        x = self.x
        y = self.y
        z = self.z
        Px = self.Px
        Py = self.Py 

        # Setting up x-matrix
        N = len(x)
        Px = Px+1; Py = Py+1
        X = np.zeros([N, Px*Py])
        for i in range(N):
            for j in range(Px):
                for k in range(Py):
                    X[i,Py*j+k] = x[i]**j*y[i]**k

        # Minimization using gradient descent
        beta = np.random.randn(Px*Py, 1)
        beta = beta[:,0]

        for iter in tqdm(range(niter)):
            e = y - X.dot(beta)                    # Absolute error
            beta += η*(2*X.T.dot(e) - q*λ*np.power(abs(beta), q-1))
        
        return np.reshape(beta.flatten(), (Px,Py))
        
        
    def ridge(self, λ=0.1):
        '''Ridge regression'''
        
        x = self.x
        y = self.y
        z = self.z
        Px = self.Px
        Py = self.Py

        # Setting up x-matrix
        N = len(x)
        Px = Px+1; Py = Py+1
        X = np.zeros([N, Px*Py])
        for i in range(N):
            for j in range(Px):
                for k in range(Py):
                    X[i,Py*j+k] = x[i]**j*y[i]**k

        # Calculating beta-vector
        beta = np.linalg.inv(X.T.dot(X)+λ*np.eye(Px*Py)).dot(X.T).dot(z)
        return np.reshape(beta.flatten(), (Px,Py))
        
        
    def lasso(self, λ=0.1, η=0.0001, niter=1000000):
        '''Lasso regression'''
        return Reg_2D.reg_q(self, 1, λ, η, niter)


def polyval(x, y, beta):
    '''Polyval 2D'''
    
    if isinstance(x, np.ndarray):
        z = np.zeros(shape=x.shape)
    else:
        z = 0
        
    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            z += beta[i,j]*np.multiply(np.power(x,i), np.power(y,j))
    return z
    
    

if __name__ == '__main__':

    # --- Simple running example ---
    N  = 100        # Number of points
    D  = 2          # Dimension
    Px = 5          # Polynomial order in x-direction
    Py = 5          # Polynomial order in y-direction

    noise = normal(0,0.1,N)

    x = uniform(0,1,N)
    y = uniform(0,1,N)
    z = FrankeFunction(x, y) + noise
    
    order5 = Reg_2D(x, y, z, Px, Py)
    
    beta_ols = order5.ols()
    beta_ridge = order5.ridge()
    beta_lasso = order5.lasso()
    
    #plt.imshow(beta_ols)
    #plt.show()
    
    #plt.imshow(beta_ridge)
    #plt.show()
    
    #plt.imshow(beta_lasso)
    #plt.show()
    
    
    x_vals = y_vals = np.linspace(0,1,1000)
    X_vals, Y_vals = np.meshgrid(x_vals, y_vals)
    predict = polyval(X_vals, Y_vals, beta_ols)
    
    # PLOT
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    
    #Plot the surface. 
    surf = ax.plot_surface(X_vals,Y_vals,predict,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    surf1 = ax.plot_surface(X_vals,Y_vals,FrankeFunction(X_vals, Y_vals),cmap=cm.coolwarm,linewidth=0,antialiased=False)

    #Customize the z axis. 
    ax.set_zlim(-0.10,1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    #Add a color bar which maps values to colors. 
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    
    # 1D plot
    y_const = 0.5
    z = FrankeFunction(x, y_const) + noise
    
    plt.plot(x, z, '.', label='Points')
    plt.plot(x_vals, polyval(x_vals, y_const, beta_ols), label='Fitted')
    plt.plot(x_vals, FrankeFunction(x_vals, y_const), label='Franke')
    plt.legend(loc='best')
    plt.show()
