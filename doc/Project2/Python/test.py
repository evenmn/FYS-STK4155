import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data, annot=True, fmt='.1f')

ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))

plt.show()
