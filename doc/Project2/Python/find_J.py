import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from ising_data import *
from regression import *

# define Ising model params
L = 40         # System size
N = 100000      # Number of states
T = 1000         # Temperature

# create random Ising states
states = produce_states([N, L], T)

X=np.einsum('...i,...j->...ij', states, states) # Outer product of each state
shape=X.shape
X=X.reshape((shape[0],shape[1]*shape[2]))       # Flatten along 0/1-axis

# calculate Ising energies
J = generate_J(L)
E = ising_energies(states,J)

n = 1000                                        # Number of states
λ = 0.001

'''
# Self-implemented
model = Reg(X[:n], E[:n])
#J_ols = model.ols()
J_ridge = model.ridge(λ)
J_lasso = model.lasso(λ)
'''

# Scikit learn
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
    print('MSE: ', (J_eval-J.flatten()).T.dot(J_eval-J.flatten())/L**2)
    #print('R2: ', (J_eval+1).dot(J_eval+1)/(J_eval+1).dot(J_eval+1))
    
    plt.imshow(J_eval.reshape(40,40))
    plt.title(J_)
    plt.colorbar()
    plt.show()
'''
    
# Using multilayer 
import neural_network as nn
n = 129900
W, b = nn.multilayer(states[:n], E[:n], T, np.array([10]))
print(nn.recall_multilayer(states[n:], W, b))
print(E[n:])
'''