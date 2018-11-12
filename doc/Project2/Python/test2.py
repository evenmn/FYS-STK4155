import numpy as np
import neural_network as nn
from activation import *
from optimization import *

# Data set
X_train = np.array([...])         # Inputs
t_train = np.array([...])         # Targets
X_test = np.array([...])          # Inputs
t_test = np.array([...])          # Targets

# Parameters
T = 10          # Number of iterations
h = [10,10]     # Number of hidden nodes
eta = 1e-4      # Learning rate
lamb = 1e-4     # Regularization parameter

# Activation functions
f1 = logistic
f2 = Leaky_ReLU
opt = SGD


obj = nn.NeuralNetwork(X_train, t_train, T, h, eta, lamb, f1, f2, opt)

obj.solver()                                      # Obtain optimal weights
y_train = obj.recall(X_train)                     # Recall training phase
y_test = obj.recall(X_test)                       # Recall test phase

Error = error_estimator(t_test, y_test)           # Error estimation
