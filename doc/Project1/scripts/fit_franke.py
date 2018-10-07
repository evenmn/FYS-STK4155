import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from franke import FrankeFunction

from resampling import *
from regression_2D import *
from regression_scikit import *
from error_tools import *


# === Constants ===
N = 1000                        # Number of sampling points
λ = 1e-5                       # Penalty
η = 0.0001                      # Learning rate
σ = 0.1                         # Standard deviation used in noise
niter = 1e5                   # Number of iterations used in Gradient Descent

noise = normal(0,σ*σ,N)         # Noise


# === Generate sampling points ===
x = uniform(0,1,N)
y = uniform(0,1,N)
z = FrankeFunction(x, y) #+ noise


'''
# === Calculating Confidence Intervals (CI) ===
order5 = Reg_2D(x, y, z, Px=5, Py=5)

avg_z, var_z, std_z = bootstrap(z)
X = order5.set_up_X()*var_z

var_beta = np.linalg.inv(X.T.dot(X))*var_z
var_beta = np.diag(var_beta)
'''

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
beta_ols_test[0,0] = beta_ols[0,0]
beta_ridge_test[0,0] = beta_ridge[0,0]
beta_lasso_test[0,0] = beta_lasso[0,0]

betas = ["beta_ols_test", "beta_ols", "beta_ridge_test", "beta_ridge", \
         "beta_lasso_test", "beta_lasso", "beta_ridge_test", "beta_ridge2"]

for beta in betas:
    beta_mat = eval(beta)
    
    #print('Confidence interval')
    #print(beta_mat.flatten()-var_beta, beta_mat.flatten()+var_beta)

    fig = plt.figure()
    plt.imshow(beta_mat, cmap=cm.coolwarm)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    #plt.savefig("../plots/{}_visualize.png".format(beta))

    plot_3D(beta_mat, show_plot=False)
    
    print("\n---{}---".format(beta))
    print("MSE: ", MSE(x, y, z, beta_mat))
    print("R2: ", R2(x, y, z, beta_mat))
plt.show()

print("OLS K=10: ", k_fold(x, y, z, K=10, method='ols'))
print("Ridge K=10: ", k_fold(x, y, z, K=10, method='ridge'))
print("Lasso K=10: ", k_fold(x, y, z, K=10, method='lasso'))
print("RidgeGD K=10: ", k_fold(x, y, z, K=10, method='ridgeGD'))

stop
# === lambda vs R2 ===
lambda_list = []
R2_ridge = []
R2_lasso = []
for i in np.linspace(-8,2,100):
    lambda_list.append(10**i)
    beta_ = order5.ridge(lambda_list[-1])
    R2_ridge.append(R2(x, y, z, beta_))
    
    beta__ = order5.lasso(lambda_list[-1], η=η, niter=niter)
    R2_lasso.append(R2(x, y, z, beta__))
    
    
label_size = {"size":"14"}
plt.semilogx(lambda_list, R2_ridge, label='Ridge', linewidth=2)
plt.semilogx(lambda_list, R2_lasso, label='Lasso', linewidth=2)
plt.xlabel('$\lambda$', **label_size)
plt.ylabel('$R^2$-score', **label_size)
plt.legend(loc='best')
plt.grid()
#plt.savefig('../plots/lambda_R2score.png')
plt.show()


# === noise vs R2 ===
R2_ols = []
R2_ridge = []
R2_lasso = []
var = []
for i in np.linspace(-6,-0.7, 100):
    var.append(10**i)
    noise = normal(0,var[-1],N)         # Noise
    z = FrankeFunction(x, y) + noise
    
    order5 = Reg_2D(x, y, z, Px=5, Py=5)
    beta_ols = order5.ols()
    beta_ridge = order5.ridge(λ)
    beta_lasso = order5.lasso(λ, η, niter)
    
    R2_ols.append(R2(x, y, z, beta_ols))
    R2_ridge.append(R2(x, y, z, beta_ridge))
    R2_lasso.append(R2(x, y, z, beta_lasso))
    
    
plt.semilogx(var, R2_ols, label='OLS', linewidth=2)
plt.semilogx(var, R2_ridge, label='Ridge', linewidth=2)
plt.semilogx(var, R2_lasso, label='Lasso', linewidth=2)
plt.xlabel('$\sigma^2$ in noise', **label_size)
plt.ylabel('$R^2$-score', **label_size)
plt.legend(loc='best')
plt.grid()
#plt.savefig('../plots/var_R2score.png')
plt.show()
