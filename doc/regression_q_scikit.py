import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from regression_q import reg_q, f

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Variables
npoints = 100
x_max  = 2
degree = 2
λ      = 0.5

# Points
x = x_max*np.random.rand(npoints,1)
y = 4+3*x+np.random.randn(npoints,1)

# x-axis
x_plot = np.linspace(0, x_max, 1000)
X_plot = x_plot[:, np.newaxis]



## --- y given by scikit learn ---
# Ridge
model_ridge = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=λ))
model_ridge.fit(x, y)
y_ridge = model_ridge.predict(X_plot)

# Lasso
model_lasso = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=λ))
model_lasso.fit(x, y)
y_lasso = model_lasso.predict(X_plot)



## --- y given by self-built ---
y_Ridge = f(x_plot, reg_q(x, y, degree, 2, λ=λ))
y_Lasso = f(x_plot, reg_q(x, y, degree, 1, λ=λ))



## --- Plot ---
# Ridge
plt.plot(x, y, '.', label='Points')
plt.title('Ridge, $\lambda$={}'.format(λ))
plt.plot(x_plot, y_ridge, label='ridge')
plt.plot(x_plot, y_Ridge, label='Ridge')
plt.legend(loc='best')
plt.grid()
plt.show()

# Lasso
plt.plot(x, y, '.', label='Points')
plt.title('Lasso, $\lambda$={}'.format(λ))
plt.plot(x_plot, y_lasso, label='lasso')
plt.plot(x_plot, y_Lasso, label='Lasso')
plt.legend(loc='best')
plt.grid()
plt.show()


## --- Mean square error estimation ---
# Ridge
diff_Ridge  = y - f(x, reg_q(x, y, degree, 2, λ=λ))
error_Ridge = diff_Ridge.T.dot(diff_Ridge)
diff_ridge  = y - model_ridge.predict(x)
error_ridge = diff_ridge.T.dot(diff_ridge)

# Lasso
diff_Lasso  = y - f(x, reg_q(x, y, degree, 1, λ=λ))
error_Lasso = diff_Lasso.T.dot(diff_Lasso)
diff_lasso  = y - model_lasso.predict(x)
error_lasso = diff_lasso.T.dot(diff_lasso)

print('Error of Ridge regression tool: ', error_Ridge[0,0])
print('Error of ridge regression tool: ', error_ridge[0,0])
print('Error of Lasso regression tool: ', error_Lasso[0,0])
print('Error of lasso regression tool: ', error_lasso)



## --- R^2 score function estimation ---
dev = (y-np.mean(y)).T.dot(y-np.mean(y))[0,0]
# Ridge
R2_Ridge = 1 - error_Ridge/dev
R2_ridge = 1 - error_ridge/dev
print('R² error function, Ridge: ', R2_Ridge[0,0])
print('R² error function, ridge: ', R2_ridge[0,0])

# Ridge
R2_Lasso = 1 - error_Lasso/dev
R2_lasso = 1 - error_lasso/dev
print('R² error function, Lasso: ', R2_Lasso[0,0])
print('R² error function, lasso: ', R2_lasso)
