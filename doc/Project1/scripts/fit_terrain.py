import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from regression_2D import *

# Matplotlib stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Load the terrain
terrain = imread('../data/s09_e116_1arc_v3.tif')

# 
Px = Py = 5

x = np.linspace(0, terrain.shape[1]-1, terrain.shape[1])
y = np.linspace(0, terrain.shape[1]-1, terrain.shape[0])

X, Y = np.meshgrid(x, y)

#order5 = Reg_2D(X, Y, terrain, Px, Py)
#beta_ols = order5.ols()

#stop

# Show the terrain
plt.figure()
plt.title('Terrain over Mount Everest')
plt.imshow(terrain)#, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# PLOT

fig = plt.figure() 
ax = fig.gca(projection='3d')

#Plot the surface. 
surf = ax.plot_surface(X,Y,terrain,cmap=cm.coolwarm,linewidth=0,antialiased=False)

#Customize the z axis. 
ax.set_zlim(0,3000)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#Add a color bar which maps values to colors. 
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
