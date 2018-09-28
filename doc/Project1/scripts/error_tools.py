import numpy as np
from regression_2D import polyval

# Matplotlib stuff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def MSE(x, y, z, beta):
    '''Mean Square Error'''
    e = z - polyval(x, y, beta)
    return e.T.dot(e)/len(x)
    
    
def R2(x, y, z, beta):
    '''R2 score function'''
    e = z - polyval(x, y, beta)
    f = y - np.average(y)
    
    return 1 - e.T.dot(e)/f.T.dot(f)
    
    
def plot_3D(beta, show_plot=True):
    '''Plot 3D'''
    
    x = y = np.linspace(0,1,1000)
    X, Y = np.meshgrid(x, y)
    Z = polyval(X, Y, beta)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #Plot the surface. 
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)

    #Customize the z axis. 
    ax.set_zlim(-0.10,1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    #Add a color bar which maps values to colors. 
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show(show_plot)
