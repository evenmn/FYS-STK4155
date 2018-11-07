import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from ising_data import *
from regression import *
from resampling import k_fold, k_fold_linreg
from error_tools import *

# define Ising model params
L = 40         # System size
N = 10000      # Number of states
T = 1000         # Temperature
path = '../plots/'

# Generate random Ising states
states = produce_states([N, L], T)

X=np.einsum('...i,...j->...ij', states, states) # Outer product of each state
shape=X.shape
X=X.reshape((shape[0],shape[1]*shape[2]))       # Flatten along 0/1-axis

# Calculate corresponding Ising energies
J = generate_J(L)
E = ising_energies(states,J)

n = int(N*0.8)                                  # Number of training states

'''
R2_train = np.zeros((12, 3))
R2_test = np.zeros((12, 3))

lambdas = np.logspace(-6,5,12)

i = 0
for λ in lambdas:

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

    j = 0
    for J_ in J_list:
        J_eval = eval(J_)
        R2_train[i,j] = R2_linreg(X[:n], J_eval, E[:n])
        R2_test[i,j] = R2_linreg(X[n:], J_eval, E[n:])
        
        print('\n--- ', J_, ' ---')
        #print('MSE_train: ', MSE_linreg(X[:n], J_eval, E[:n]))
        #print('MSE_test: ', MSE_linreg(X[n:], J_eval, E[n:]))
        print('R2_train: ', R2_train[i,j])
        print('R2_test: ', R2_test[i,j])
        
        
        MSE_train_kfold, MSE_test_kfold, R2_train_kfold, R2_test_kfold = k_fold_linreg(X, E, λ, K=10, method=J_)
        print('MSE_train_kfold: ', MSE_train_kfold)
        print('MSE_test_kfold: ', MSE_test_kfold)
        print('R2_train_kfold: ', R2_train_kfold)
        print('R2_test_kfold: ', R2_test_kfold)
        
        
        J_eval = J_eval.reshape(L,L)
        plt.figure()
        plt.imshow(J_eval)
        plt.title(J_ + ' λ = {}'.format(λ))
        cbar = plt.colorbar(cmap='coolwarm')
        cbar.ax.tick_params(labelsize=22)
        plt.savefig(path + J_ + '_lambda_{}.png'.format(int(-np.log10(λ))))
        
        j += 1
    i += 1

plt.semilogx(lambdas, R2_train[:,0], label='OLS (train)')
plt.semilogx(lambdas, R2_test[:,0], '--', label='OLS (test)')
plt.semilogx(lambdas, R2_train[:,1], label='Ridge (train)')
plt.semilogx(lambdas, R2_test[:,1], '--', label='Ridge (test)')
plt.semilogx(lambdas, R2_train[:,2], label='Lasso (train)')
plt.semilogx(lambdas, R2_test[:,2], '--', label='Lasso (test)')
plt.legend(loc='best')
plt.xlabel('$\lambda$',fontsize=16)
plt.ylabel('R$^2$-score',fontsize=16)
plt.grid()
plt.show()
'''
    
# Neural network
import neural_network as nn
'''
# === Linear ===
W = nn.linear(X[:n], E[:n], 50)
E_tilde_train = nn.recall_linear(X[:n], W)
E_tilde_test = nn.recall_linear(X[n:], W)

print('MSE_train: ', MSE(E_tilde_train, E[:n]))
print('MSE_test: ', MSE(E_tilde_test, E[n:]))
print('R2_train: ', R2(E_tilde_train, E[:n]))
print('R2_test: ', R2(E_tilde_test, E[n:]))


MSE_train_kfold, MSE_test_kfold, R2_train_kfold, R2_test_kfold = k_fold(X, E, K=10)
print('MSE_train_kfold: ', MSE_train_kfold)
print('MSE_test_kfold: ', MSE_test_kfold)
print('R2_train_kfold: ', R2_train_kfold)
print('R2_test_kfold: ', R2_test_kfold)
'''

# === Nonlinear ===
W1, W2, b1, b2 = nn.nonlinear(X[:n], E[:n], 50, 100)
E_tilde_train = nn.recall_nonlinear(X[:n], W1, W2, b1, b2)
E_tilde_test = nn.recall_nonlinear(X[n:], W1, W2, b1, b2)

print('MSE_train: ', MSE(E_tilde_train, E[:n]))
print('MSE_test: ', MSE(E_tilde_test, E[n:]))
print('R2_train: ', R2(E_tilde_train, E[:n]))
print('R2_test: ', R2(E_tilde_test, E[n:]))


'''
# === Multilayer ===
W, b = nn.multilayer(X[:n], E[:n], 500, [1])
E_tilde_train = nn.recall_multilayer(X[:n], W, b)
E_tilde_test = nn.recall_multilayer(X[n:], W, b)

print('MSE_train: ', MSE(E_tilde_train, E[:n]))
print('MSE_test: ', MSE(E_tilde_test, E[n:]))
print('R2_train: ', R2(E_tilde_train, E[:n]))
print('R2_test: ', R2(E_tilde_test, E[n:]))


MSE_train_kfold, MSE_test_kfold, R2_train_kfold, R2_test_kfold = k_fold(X, E, K=10)
print('MSE_train_kfold: ', MSE_train_kfold)
print('MSE_test_kfold: ', MSE_test_kfold)
print('R2_train_kfold: ', R2_train_kfold)
print('R2_test_kfold: ', R2_test_kfold)
'''
