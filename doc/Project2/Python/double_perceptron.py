#!/usr/bin/python

'''
The perc function takes the inputs X and the targets t as arguments, 
which are lists of 2-lists. It also takes the learning rate eta, and
returns the best-fitting weights. 

Length input           : 2
Length output          : 2
Number of layers       : 2
Number of hidden nodes : 2
'''

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return (1 + np.exp(-x))**(-1)
    
def sig_der(x):
    return x*(1 - x)

# Perceptron function
def perc(X, t, eta, vector = True):
    # Weights
    #W1 = [[w1, w3], [w2, w4]]
    W1 = 2*np.random.random([2,2]) - 1

    #W2 = [[w5, w7], [w6, w8]]
    W2 = 2*np.random.random([2,2]) - 1

    b1 = np.random.random()
    b2 = np.random.random()

    # Training
    for iter in range(200000):
        for i in range(len(X)):
            # Forward propagation
            net_h = np.dot(X[i], W1) + [b1, b1]
            out_h = sigmoid(net_h)

            net_o = np.dot(out_h, W2) + [b2, b2]
            out_o = sigmoid(net_o)
            print("Outputs: ", out_o)

            # Total error
            E_TOT = sum(0.5*((t[i] - out_o)**2))
            print("Error:   ", E_TOT, "\n")

            # BACKWARD PROPAGATION
            
            if vector:
                # Vectorized
                deltao = -(t[i] - out_o)*out_o*(1 - out_o)
                deltaE = np.outer(deltao, out_h)

                W2_new = W2 - eta * np.transpose(deltaE)
                
                # Update w1 and w2
                dEodouth1 = np.multiply((out_o - t[i]), np.multiply(sig_der(out_o), W2[0]))

                w1w2_new = W1[:,0] - eta * sum(dEodouth1) * np.dot(sig_der(out_h[0]), X[i])
                w1_new = w1w2_new[0]
                w2_new = w1w2_new[1]

                # Update w3 and w4
                dEodouth2 = np.multiply((out_o - t[i]), np.multiply(sig_der(out_o)[::-1], W2[1]))
                
                w3w4_new = W1[:,1] - eta * sum(dEodouth2) * np.dot(sig_der(out_h[1]), X[i])
                w3_new = w3w4_new[0]
                w4_new = w3w4_new[1]
            '''
            else:
                # Scalar
                deltao1 = -(t[0] - out_o[0])*out_o[0]*(1 - out_o[0])
                deltao2 = -(t[1] - out_o[1])*out_o[1]*(1 - out_o[1])

                deltaEdeltaW5 = deltao1 * out_h[0]
                deltaEdeltaW6 = deltao1 * out_h[1]
                deltaEdeltaW7 = deltao2 * out_h[0]
                deltaEdeltaW8 = deltao2 * out_h[1]

                W5_new = W2[0,0] - eta * deltaEdeltaW5
                W6_new = W2[1,0] - eta * deltaEdeltaW6
                W7_new = W2[0,1] - eta * deltaEdeltaW7
                W8_new = W2[1,1] - eta * deltaEdeltaW8

                # Update w1
                dEo1douto1 = -(t[0] - out_o[0])
                douto1dneto1 = out_o[0]*(1 - out_o[0])
                dEo1dneto1 = dEo1douto1 * douto1dneto1 

                dneto1douth1 = W2[0,0]
                dEo1douth1 = dEo1dneto1 * dneto1douth1

                dEo2douto1 = -(t[1] - out_o[1])
                douto1dneto1 = out_o[1]*(1 - out_o[1])
                dEo2dneto1 = dEo2douto1 * douto1dneto1

                dneto1douth1 = W2[0,1]
                dEo2douth1 = dEo2dneto1 * dneto1douth1

                dEdouth1 = dEo1douth1 + dEo2douth1

                douth1dneth1 = out_h[0]*(1 - out_h[0])
                dneth1dw1 = X[0]

                dEdw1 = dEdouth1 * douth1dneth1 * dneth1dw1
                w1_new = W1[0,0] - eta * dEdw1

                # Update w2
                dneth1dw2 = X[1]

                dEdw2 = dEdouth1 * douth1dneth1 * dneth1dw2
                w2_new = W1[1,0] - eta * dEdw2

                # Update w3
                dEo1douto2 = -(t[1] - out_o[1])
                douto2dneto2 = out_o[0]*(1 - out_o[0])
                dEo1dneto2 = dEo1douto2 * douto2dneto2

                dneto2douth2 = W2[1,1]
                dEo1douth2 = dEo1dneto2 * dneto2douth2

                dEo2douto2 = -(t[0] - out_o[0])
                douto2dneto2 = out_o[1]*(1 - out_o[1])
                dEo2dneto2 = dEo2douto2 * douto2dneto2

                dneto2douth2 = W2[1,0]
                dEo2douth2 = dEo2dneto2 * dneto2douth2

                dEdouth2 = dEo1douth2 + dEo2douth2

                douth2dneth2 = out_h[1]*(1 - out_h[1])
                dneth2dw3 = X[0]

                dEdw3 = dEdouth2 * douth2dneth2 * dneth2dw3
                w3_new = W1[0,1] - eta * dEdw3

                # update w4
                dneth2dw4 = X[1]

                dEdw4 = dEdouth2 * douth2dneth2 * dneth2dw4
                w4_new = W1[1,1] - eta * dEdw4
            '''

            W1 = np.array([[w1_new, w3_new], [w2_new, w4_new]])
            W2 = W2_new
    
    return W1, W2

if __name__ == '__main__':
    pass

# inputs and targets
X = [[0,0], [0,1], [1,0], [1,1], [0,2], [2,0], [2,1], [0,3], [2,3], [3,3]]
t = [[0.0, 0.0], [0.0, 0.1], [0.0, 0.1], [0.1, 0.2], [0.0, 0.2], [0.0, 0.2], [0.2, 0.3], [0.0, 0.3], [0.6, 0.5], [0.9, 0.6]]

W1, W2 = perc(X, t, 0.2)

test = [[1,2], [2,2], [1,3], [3,1], [2,3], [3,2], [3,3]]
# Should give outputs [[2,3], [4,4], [3,4], [3,4], [6,5], [6,5], [9,6]]/10

b1 = np.random.random()
b2 = np.random.random()

for x in test:
    net_h = np.dot(x, W1) + [b1, b1]
    out_h = sigmoid(net_h)

    net_o = np.dot(out_h, W2) + [b2, b2]
    out_o = sigmoid(net_o)
    print("Test output: ", out_o)
