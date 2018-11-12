import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from Ising_1D import *
from regression import *
from resampling import k_fold, k_fold_linreg
from error_tools import *
from activation import *

# define Ising model params
L = 40              # System size
N = 10000           # Number of states
T = 1000            # Temperature
path = '../plots/'  # Path to plots
method = 1          # 0 is linear regression
                    # 1 is neural network

# Generate random Ising states
states = produce_states([N, L], T)

X=np.einsum('...i,...j->...ij', states, states) # Outer product of each state
shape=X.shape
X=X.reshape((shape[0],shape[1]*shape[2]))       # Flatten along 0/1-axis

# Calculate corresponding Ising energies
J = generate_J(L)
E = ising_energies(states,J)

n = int(N*0.8)                                  # Number of training states
X_train = X[:n]
X_test = X[n:]
E_train = E[:n]
E_test = E[n:]

if method == 0:
    R2_train = np.zeros((12, 3))
    R2_test = np.zeros((12, 3))

    lambdas = np.logspace(-6,5,12)

    i = 0
    for λ in lambdas:

        # Linear regression
        ols=linear_model.LinearRegression()
        ols.fit(X_train, E_train)
        J_ols = ols.coef_

        ridge=linear_model.Ridge()
        ridge.set_params(alpha=λ)
        ridge.fit(X_train, E_train)
        J_ridge = ridge.coef_

        lasso=linear_model.Lasso()
        lasso.set_params(alpha=λ)
        lasso.fit(X_train, E_train)
        J_lasso = lasso.coef_

        J_list = ['J_ols', 'J_ridge', 'J_lasso']

        j = 0
        for J_ in J_list:
            J_eval = eval(J_)
            R2_train[i,j] = R2_linreg(X_train, J_eval, E_train)
            R2_test[i,j] = R2_linreg(X_test, J_eval, E_test)
            
            print('\n--- ', J_, ' ---')
            print('MSE_train: ', MSE_linreg(X_train, J_eval, E_train))
            print('MSE_test: ', MSE_linreg(X_test, J_eval, E_test))
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

elif method == 1:    
    # Neural network
    import neural_network as nn
    h_list = [5,10]
    f2_list = [ELU]
    for f2 in f2_list:
        for h in h_list:
            print('=== {} === {} ==='.format(f2, h))
            #h = [5,5]                         # Number of hidden nodes
            T = 5                         # Number of iterations
            eta = 0.0001
            #f2=Leaky_ReLU

            obj = nn.NeuralNetwork(X_train, E_train, T, h, eta, f2=f2)     # Define object
            obj.solver()                                            # Obtain optimal weights
            E_tilde_train = obj.recall(X_train)                     # Recall training energy
            E_tilde_test = obj.recall(X_test)                       # Recall test energy

            print('MSE_train: ', MSE(E_tilde_train, E_train))
            print('MSE_test: ', MSE(E_tilde_test, E_test))
            print('R2_train: ', R2(E_tilde_train, E_train))
            print('R2_test: ', R2(E_tilde_test, E_test))

            MSE_train_kfold, MSE_test_kfold, R2_train_kfold, R2_test_kfold = k_fold(X, E, T, h, f2, eta, K=10)
            print('MSE_train_kfold: ', MSE_train_kfold)
            print('MSE_test_kfold: ', MSE_test_kfold)
            print('R2_train_kfold: ', R2_train_kfold)
            print('R2_test_kfold: ', R2_test_kfold)
