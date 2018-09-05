import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from regression import reg, f

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Variables
x_max = 3
degree = 3

# Points
x = x_max*np.random.rand(50,1)
y = 4+3*x+np.random.randn(50,1)

# x-axis
x_plot = np.linspace(0, x_max, 100)
X_plot = x_plot[:, np.newaxis]

# y given by scikit learn
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(x, y)
y_plot = model.predict(X_plot)

# y given by self-built
y_self = f(x_plot, reg(x, y, degree))

# Plot
plt.plot(x, y, '.', label='Points')
plt.plot(x_plot, y_plot, label='Scikit-learn')
plt.plot(x_plot, y_self, label='Self-built')
plt.legend(loc='best')
plt.grid()
plt.show()

# Mean square error estimation
error_self = np.sum((y - f(x, reg(x, y, 2)))**2)
error_scikit = np.sum((y - model.predict(x))**2)

#diff = y-f(x, reg(x, y, 2))
#print(np.dot(np.transpose(diff), diff))            # alt

print('Error of self-built regression tool: ', error_self)
print('Error of scikit regression tool: ', error_scikit)

# R^2 score function estimation
dev = np.dot(np.transpose(y-np.mean(y)), y-np.mean(y))[0,0]
R2_self = 1 - error_self/dev
R2_scikit = 1 - error_scikit/dev

print('R² error function, self-built: ', R2_self)
print('R² error function, scikit: ', R2_scikit)
