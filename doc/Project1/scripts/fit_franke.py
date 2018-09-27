import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from franke import FrankeFunction
from regression_2D import *
from regression_scikit import *
from error_tools import *


# === Constants ===
N = 100                         # Number of sampling points
λ = 1e-1                        # Penalty
η = 0.0001                      # Learning rate
σ2 = 0.1                        # Variance used in noise
niter = 1e5                     # Number of iterations used in Gradient Descent

noise = normal(0,σ2,N)          # Noise


# === Generate sampling points ===
x = uniform(0,1,N)
y = uniform(0,1,N)
z = FrankeFunction(x, y) + noise


# === Call self-built regression functions ===
order5 = Reg_2D(x, y, z, Px=5, Py=5)

beta_ols = order5.ols()
beta_ridge = order5.ridge(λ)
print("\n Doing Lasso regression..."); beta_lasso = order5.lasso(λ, η, niter)
print("\n Doing Ridge regression..."); beta_ridge2 = order5.reg_q(2, λ, η, niter)


# === Call scikit regression functions ===
order5_scikit = Reg_scikit(x, y, z, Px=5, Py=5)

beta_ols_test = order5_scikit.ols()
beta_lasso_test = order5_scikit.lasso(λ)
beta_ridge_test = order5_scikit.ridge(λ)


# === PLOT ===
beta_ols_test_new = beta_ols_test
beta_ols_test_new[0,0] = 0

betas = ["beta_ols_test_new", "beta_ols", "beta_ridge_test", "beta_ridge", \
         "beta_lasso_test", "beta_lasso", "beta_ridge_test", "beta_ridge2"]
'''
for beta in betas:
    beta_mat = eval(beta)

    fig = plt.figure()
    plt.imshow(beta_mat)
    plt.savefig("../plots/{}_visualize.png".format(beta))

    #plot_3D(beta_mat, show_plot=False)
    
    print("\n---{}---".format(beta))
    print("MSE: ", MSE(x, y, z, beta_mat))
    print("R2: ", R2(x, y, z, beta_mat))
plt.show()
'''

# === lambda vs R2 ===
lambda_list = [10**i for i in np.linspace(-8,5)]
R2_ridge = []
R2_lasso = []

for lamb in lambda_list:
    beta_ = order5.ridge(lamb)
    R2_ridge.append(R2(x, y, z, beta_))
    
    #beta__ = order5.lasso(lamb)
    #R2_lasso.append(R2(x, y, z, beta__))
    
plt.semilogx(lambda_list, R2_ridge, label='Ridge')
#plt.semilogx(lambda_list, R2_lasso, label='Lasso')
plt.xlabel('$\lambda$')
plt.ylabel('$R^2$-score')
plt.legend(loc='best')
plt.show()
