import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('../data/s09_e116_1arc_v3.tif')

print(terrain1.shape)
# Show the terrain
plt.figure()
plt.title('Terrain over Mount Everest')
plt.imshow(terrain1)#, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
