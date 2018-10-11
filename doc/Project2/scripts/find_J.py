import numpy as np
import matplotlib.pyplot as plt
from ising_data import *
from regression import *

# define Ising model params
L=4         # system size
N=1000000   # Number of states

# create random Ising states
states = np.random.choice([-1, 1], size=(N, L))
X = np.multiply(states[:,1:], states[:,:-1])

# calculate Ising energies
E = ising_energies(states, L)

model = Reg(X, E)

J_ols = model.ols()
J_ridge = model.ridge()
J_lasso = model.lasso()

J_list = ['J_ols', 'J_ridge', 'J_lasso']

for J in J_list:
    J_eval = eval(J)
    
    print('\n--- ', J, ' ---')
    print(J_eval)
    print('MSE: ', (J_eval+1).dot(J_eval+1)/len(L))
    print('R2: ', (J_eval+1).dot(J_eval+1)/(J_eval+1).dot(J_eval+1)
