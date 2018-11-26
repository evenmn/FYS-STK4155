import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import neural_network as nn
import librosa
import pandas as pd
from activation import *
from optimization import *

# ==== REPROGRESS DATA ==== 

def class2binary(class_):
    if class_ == "air_conditioner":
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif class_ == "car_horn":
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif class_ == "children_playing":
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif class_ == "dog_bark":
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif class_ == "drilling":
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif class_ == "engine_idling":
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif class_ == "gun_shot":
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif class_ == "jackhammer":
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif class_ == "siren":
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif class_ == "street_music":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


train = pd.read_csv("../data/train.csv")

N = len(train) -2
data_train = np.zeros((N, 88200))
t_train = np.zeros((N, 10))


i = 0
j = 0
count = 0
while i < N:
    print(j)
    
    if j != 1986 and j != 5312:
        x, sr = librosa.load('../data/Train/' + str(train.ID[j]) + '.wav')
        target = class2binary(train.Class[i])
        
        data_train[i,:len(x)] = x
        t_train[i] = np.array(target)
        
        i += 1
    j += 1


# ====  REGRESSION ==== 

# Self
h = [10]

T = 100
obj = nn.NeuralNetwork(data_train, t_train, T, 'random', h, eta=0.001, lamb=0.001, f1=logistic, f2=Leaky_ReLU, opt=GD)
obj.solver()                                      # Obtain optimal weights

y = obj.recall(data_train)                     # Recall training phase

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
    
print(Accuracy(y_new, t_train))


# Logistic scikit
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(data_train, t_train)
print(clf.score(data_train, t_train))


# Neural network scikit
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(data_train, t_train)
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
print(clf.score(data_train, t_train))
