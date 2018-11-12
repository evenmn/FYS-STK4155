import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

train = np.loadtxt('../data/train.dat')
test = np.loadtxt('../data/test.dat')
critical = np.loadtxt('../data/critical.dat')

logspace = np.logspace(-6,3,10)
arange = np.arange(10)
size=16

# Heatmap
ax = sns.heatmap(train, annot=True, fmt='.2f', cmap="PuRd", cbar=False, annot_kws={"size": size})

ax.set_xticks(arange)
ax.set_yticks(arange)

ax.set_xticklabels(logspace, fontsize=size)
ax.set_yticklabels(logspace, fontsize=size)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Train", fontsize=size)
plt.xlabel("$\eta$", fontsize=size)
plt.ylabel("$\lambda$", fontsize=size)
plt.show()



ax = sns.heatmap(test, annot=True, fmt='.2f', cmap="PuRd", cbar=False, annot_kws={"size": size})

ax.set_xticks(arange)
ax.set_yticks(arange)

ax.set_xticklabels(logspace, fontsize=size)
ax.set_yticklabels(logspace, fontsize=size)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Test", fontsize=size)
plt.xlabel("$\eta$", fontsize=size)
plt.ylabel("$\lambda$", fontsize=size)
plt.show()



ax = sns.heatmap(critical, annot=True, fmt='.2f', cmap="PuRd", cbar=False, annot_kws={"size": size})

ax.set_xticks(arange)
ax.set_yticks(arange)

ax.set_xticklabels(logspace, fontsize=size)
ax.set_yticklabels(logspace, fontsize=size)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Critical", fontsize=size)
plt.xlabel("$\eta$", fontsize=size)
plt.ylabel("$\lambda$", fontsize=size)
plt.show()
