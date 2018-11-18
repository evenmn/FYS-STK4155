import matplotlib.pyplot as plt
import neural_network as nn
from activation import *
from optimization import *

t1 = [1,0,0,0,0,0,0,0,0,0]
t2 = [0,1,0,0,0,0,0,0,0,0]

data = np.array([data1, data2])
t = np.array([t1, t2])

h = [10,10,10]

T = 1
obj = nn.NeuralNetwork(data, t, T, h, eta=0.0001, lamb=0.001, f1=logistic, f2=Leaky_ReLU, opt=GD)

obj.solver()                                      # Obtain optimal weights



