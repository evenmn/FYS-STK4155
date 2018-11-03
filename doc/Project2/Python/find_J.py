import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from ising_data import *
from regression import *
from resampling import k_fold
from error_tools import *

# define Ising model params
L = 40         # System size
N = 10000      # Number of states
T = 1000         # Temperature

# create random Ising states
states = produce_states([N, L], T)

X=np.einsum('...i,...j->...ij', states, states) # Outer product of each state
shape=X.shape
X=X.reshape((shape[0],shape[1]*shape[2]))       # Flatten along 0/1-axis

# calculate Ising energies
J = generate_J(L)
E = ising_energies(states,J)

n = int(4*N/5)                                  # Number of training states
λ = 0.001

'''
# Linear regression
ols=linear_model.LinearRegression()
ols.fit(X[:n], E[:n])
J_ols = ols.coef_

ridge=linear_model.Ridge()
ridge.set_params(alpha=λ)
ridge.fit(X[:n], E[:n])
J_ridge = ridge.coef_

lasso=linear_model.Lasso()
lasso.set_params(alpha=λ)
lasso.fit(X[:n], E[:n])
J_lasso = lasso.coef_

J_list = ['J_ols', 'J_ridge', 'J_lasso']

for J_ in J_list:
    J_eval = eval(J_)
    
    print('\n--- ', J_, ' ---')
    print(J_eval)
    print('MSE_train: ', MSE(X[:n], J_eval, E[:n]))
    print('MSE_test: ', MSE(X[n:], J_eval, E[n:]))
    print('R2_train: ', R2(X[:n], J_eval, E[:n]))
    print('R2_test: ', R2(X[n:], J_eval, E[n:]))
    
    MSE_train_kfold, MSE_test_kfold, R2_train_kfold, R2_test_kfold = k_fold(X, E, L, λ, K=10, method=J_)
    print('MSE_train_kfold: ', MSE_train_kfold)
    print('MSE_test_kfold: ', MSE_test_kfold)
    print('R2_train_kfold: ', R2_train_kfold)
    print('R2_test_kfold: ', R2_test_kfold)
    
    J_eval = J_eval.reshape(L,L)
    plt.imshow(J_eval)
    plt.title(J_)
    plt.colorbar()
    plt.show()
'''
    
# Neural network
import neural_network as nn
from transformation import *

W = nn.linear2(states[:n], E[:n], 20000)
E_tilde = nn.recall_linear(states[:n], W)

print(E_tilde)
print(E[:n])

