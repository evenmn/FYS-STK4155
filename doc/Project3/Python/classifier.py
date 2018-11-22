import matplotlib.pyplot as plt
import neural_network as nn
from activation import *
from optimization import *
import dask.array as da
import dask.dataframe as dd
from dask_ml.linear_model import LogisticRegression
import pandas as pd

#model = LogisticRegression()
#model.fit(data, labels)

#data = np.loadtxt('../data/data_train.dat')
t = np.loadtxt('../data/t_train.dat')
data = dd.read_csv('../data/data_train.csv')


#data = da.from_array(data, chunks=(100))
#t = da.from_array(t[:len(data)], chunks=(100))
t = t[:len(data)]
h = [1]

T = 100
obj = nn.NeuralNetwork(data, t, T, h, eta=0.001, lamb=0.001, f1=logistic, f2=Leaky_ReLU, opt=GD)
obj.solver()                                      # Obtain optimal weights

y = obj.recall(data)                     # Recall training phase

# Need to set the largest value to 1, and the rest to zero
y_new = np.zeros(y.shape)
for i in range(len(y)):
    j = np.argmax(y[i])
    y_new[i,j] = 1

def Accuracy(y, t):
    counter = 0
    for i in range(len(y)):
        if np.array_equal(y[i], t[i]):
            counter += 1
    return counter/len(y)
    
print(Accuracy(y_new, t))
