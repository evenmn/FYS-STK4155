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
    return np.reshape(beta.flatten(), (Px,Py))



def f_2D(x, y, beta):
    '''Polyval'''
    
    if isinstance(x, np.ndarray):
        z = np.zeros(shape=x.shape)
    else:
        z = 0
        
    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            z += beta[i,j]*np.multiply(np.power(x,i), np.power(y,j))
    return z
    
    

if __name__ == '__main__':
    N  = 100        # Number of points
    D  = 2          # Dimension
    Px = 5          # Polynomial order in x-direction
    Py = 5          # Polynomial order in y-direction

    noise = normal(0,0.1,N)

    x = uniform(0,1,N)
    y = uniform(0,1,N)
    z = FrankeFunction(x, y) + noise
    
    beta = reg_2D(x, y, z, Px, Py)
    
    x_vals = y_vals = np.linspace(0,1,1000)
    X_vals, Y_vals = np.meshgrid(x_vals, y_vals)
    predict = f_2D(X_vals, Y_vals, beta)
    
    # PLOT
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    
    #Plot the surface. 
    surf = ax.plot_surface(X_vals,Y_vals,predict,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    #surf = ax.plot_surface(X_vals,Y_vals,FrankeFunction(X_vals, Y_vals),cmap=cm.coolwarm,linewidth=0,antialiased=False)

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
    plt.plot(x_vals, f_2D(x_vals, y_const, beta), label='Fitted')
    plt.plot(x_vals, FrankeFunction(x_vals, y_const), label='Franke')
    plt.legend(loc='best')
    plt.show()
    
    
    
    
    # Confidence interval, beta
    print(np.var(beta))
    
    
    # Mean square error (MSE):
    MSE = (z - f_2D(x, y, beta)).T.dot(z - f_2D(x, y, beta))/N
    print(MSE)
    
    
    # R2 score function
    denominator = (y-np.mean(y)).T.dot(y-np.mean(y))
    print(1-N*MSE/denominator)
