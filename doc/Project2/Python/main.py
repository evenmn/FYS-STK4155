#!/usr/bin/python

import neural_network as nn
import numpy as np

'''
neural_network.py is an 1- or 2-layer artificial neural network which in principle 
can be used to solve any problem, and in practice can be used to solve a lot
of problems. In this main I will give some examples on how to use it. The input
are of dynamic size, but need to be given as 2D numpy arrays.

X - Input array
t - Target array
N - Number of training iterations
H - Number of hidden nodes
(eta - Learning rate, set to 0.1, but can be changed as last argument)
'''

# --- OR gate ---
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Input
t = np.array([[0], [1], [1], [1]])          # Output

W, b = nn.linear(X, t, 500)                   # Receive weights after training
net = nn.recall_linear(X, W, b)                # Use weights to get output
out = np.where(net>0.5,1,0)                 # Round to integer

print('--- OR gate ---')
print(net)
print(out,'\n')


# --- XOR gate ---
X = np.array([[0,0], [1,0], [0,1], [1,1]])
t = np.array([[0], [1], [1], [0]])

W1, W2, b1, b2 = nn.nonlinear(X, t, 5000, 5)      # Training - 10000 cycles and 5 hidden nodes
net = nn.recall_nonlinear(X, W1, W2, b1, b2)      # Using the weights to verify the result
out = np.where(net>0.5,1,0)

print('--- XOR gate ---')
print(net)
print(out,'\n')

# --- Heavy calculations ---
'''Sometimes two layers is not sufficient, and one needs to use more layers to
achieve the correct results. When call multilayer'''

X = np.array([[0,0], [1,0], [0,1], [1,1]])
t = np.array([[0], [1], [1], [0]])
h = np.array([5, 5])        #Two hidden layers of 5 elements each

W, b = nn.multilayer(X, t, 10000, h)
net = nn.recall_multilayer(X, W, b)
out = np.where(net>0.5,1,0)

print('--- XOR gate, multilayer ---')
print(net)
print(out,'\n')
