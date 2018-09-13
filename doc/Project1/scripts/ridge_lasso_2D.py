import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import uniform, normal
from franke import FrankeFunction
from OLS_2D import f_2D

# Matplotlib stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def reg_q_2D(x, y, z, Px, Py, q, λ=0.1, η=0.0001, niter=10000):
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
    

    # Minimization using gradient descent
    beta = np.random.randn(Px*Py, 1)

    for iter in tqdm(range(niter)):
        e = y - X.dot(beta).T                    # Absolute error
        beta += η*(2*X.T.dot(e.T) - q*λ*np.power(abs(beta), q-1))
    
    return np.reshape(beta.flatten(), (Px,Py))
    
    
if __name__ == "__main__":
    N  = 100        # Number of points
    D  = 2          # Dimension
    Px = 5          # Polynomial order in x-direction
    Py = 5          # Polynomial order in y-direction

    noise = normal(0,0.0,N)

    x = uniform(0,1,N)
    y = uniform(0,1,N)
    z = FrankeFunction(x, y) #+ noise
    
    beta = reg_q_2D(x, y, z, Px, Py, 1)         # Ridge
    
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
    
    # 1D plot
    y_const = 0.5
    z = FrankeFunction(x, y_const) + noise
    
    plt.plot(x, z, '.', label='Points')
    plt.plot(x_vals, f_2D(x_vals, y_const, beta), label='Fitted')
    plt.plot(x_vals, FrankeFunction(x_vals, y_const), label='Franke')
    plt.legend(loc='best')
    plt.show()
