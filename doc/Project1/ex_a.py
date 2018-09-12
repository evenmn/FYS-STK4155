import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from franke import FrankeFunction

# Matplotlib stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def reg_2D(x, y, z, Px, Py):
    '''Regression, finding coefficients beta
    
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

    # Setting up x-matrix
    Px = Px+1; Py = Py+1
    X = np.zeros([N, Px*Py])
    for i in range(N):
        for j in range(Px):
            for k in range(Py):
                X[i,Py*j+k] = x[i]**j*y[i]**k

    # Calculating beta-vector
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    return beta.flatten() #[::-1]



def f_2D(x, y, pol):
    '''Polyval'''
    return np.polynomial.polynomial.polyval2d(x, y, pol)
    
    

if __name__ == '__main__':
    N  = 100        # Number of points
    D  = 2          # Dimension
    Px = 5          # Polynomial order in x-direction
    Py = 5          # Polynomial order in y-direction

    x = uniform(0,1,N)
    y = uniform(0,1,N)
    z = FrankeFunction(x, y)
    
    beta = reg_2D(x, y, z, Px, Py)
    beta = np.reshape(beta, (Px+1,Py+1))
    
    x_vals = y_vals = np.linspace(0,1,1000)
    X_vals, Y_vals = np.meshgrid(x_vals, y_vals)
    predict = f_2D(X_vals, Y_vals, beta)
    
    # PLOT
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    
    #Plot the surface. 
    surf = ax.plot_surface(X_vals,Y_vals,predict,cmap=cm.coolwarm,linewidth=0,antialiased=False)

    #Customize the z axis. 
    ax.set_zlim(-0.10,1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #Add a color bar which maps values to colors. 
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
