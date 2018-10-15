import numpy as np
import matplotlib.pyplot as plt
from ising_data import *
from regression import *

# define Ising model params
L = 100         # System size
N = 100000   # Number of states
T = 100       # Temperature

# create random Ising states
states = produce_states([N, L], T)
X = np.multiply(states[:,1:], states[:,:-1])

# calculate Ising energies
E = ising_energies(states)

print(X)
print(E)

model = Reg(X, E)

#J_ols = model.ols()
J_ridge = model.ridge()
#J_lasso = model.lasso()

J_list = ['J_ridge'] #['J_ols', 'J_ridge', 'J_lasso']

for J in J_list:
    J_eval = eval(J)
    
    print('\n--- ', J, ' ---')
    print(J_eval)
    print('MSE: ', (J_eval+1).dot(J_eval+1)/L)
    #print('R2: ', (J_eval+1).dot(J_eval+1)/(J_eval+1).dot(J_eval+1))
