# Import packages
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

import neural_network as nn
from activation import *
from optimization import *

# Convert the data to pass it in our deep learning model
#temp = pd.read_csv('../data/temp.csv')

#X = np.array(temp.features.tolist())
#t = np.array(temp.labels.tolist())

X = np.loadtxt('../data/X_40.txt')
text_file = open('../data/t.txt', 'r')
t = []
for line in text_file:
    t.append(line)
lb = LabelEncoder()
t = np_utils.to_categorical(lb.fit_transform(t))


# Split into training and test set
N = len(X)
N_train = int(len(X)*0.8)
X_train = X[:N_train]
t_train = t[:N_train]
X_test = X[N_train:]
t_test = t[N_train:]


# Run a deep learning model and get results
method = 1

if method == 0:
    obj = nn.NeuralNetwork(X, t, T=50, h=[100,100], eta=0.001, lamb=0.001, f1=logistic, f2=ReLU, opt=GD)
    obj.solver()                                      # Obtain optimal weights

    y = obj.recall(X)                     # Recall training phase

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
    

# build model
elif method == 1:
    num_labels = t.shape[1]

    model = Sequential()

    model.add(Dense(1024, input_shape=(40,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, t_train, batch_size=32, epochs=1000, validation_data=(X_test, t_test))
    weight = model.layers[0].get_weights()
    print(weight)
    
    
elif method == 2:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf.fit(X, t)
    print(clf.score(X, t))
