import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from regression_2D import *
from time import clock

# Matplotlib stuff
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Load the terrain
terrain = imread('../data/s09_e116_1arc_v3.tif')

x = np.linspace(0, terrain.shape[1]-1, terrain.shape[1])
y = np.linspace(0, terrain.shape[1]-1, terrain.shape[0])

X, Y = np.meshgrid(x, y)

X_flatten = X.flatten()
Y_flatten = Y.flatten()
Z_flatten = terrain.flatten()

factor = 10
X_flatten = X_flatten[::factor]
Y_flatten = Y_flatten[::factor]
Z_flatten = Z_flatten[::factor]

start = clock()
order5 = Reg_2D(X_flatten, Y_flatten, Z_flatten, Px=5, Py=5)
beta_ols = order5.ridge()
end = clock()
print(end-start)
print(beta_ols)

# Show the terrain
plt.figure()
plt.title('Terrain over Mount Everest')
plt.imshow(terrain)#, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.imshow(polyval(X, Y, beta_ols))
plt.show()
stop
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
