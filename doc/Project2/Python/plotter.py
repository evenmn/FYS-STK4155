import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

one_layer = np.array([0.46172, 0.46172, 0.46172, 0.46172, 0.4575, 0.46172, 0.46172, 0.99852, 0.99852, 0.99852, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866, 0.99866])
two_layer = np.array([0.46446, 0.46438, 0.46438, 0.46438, 0.4644, 0.4644, 0.99864, 0.46438, 0.99866, 0.46188, 0.99864, 0.99864, 0.99864, 0.99864, 0.99866, 0.99864, 0.99864, 0.99864, 0.99864, 0.99864, 0.99864])

x = np.arange(len(one_layer), dtype=int)

plt.plot(x, one_layer, label='One layer')
plt.plot(x, two_layer, label='Two layers')
plt.legend(loc='best', fontsize=16)
plt.xlabel('Number of hidden nodes', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.show()
