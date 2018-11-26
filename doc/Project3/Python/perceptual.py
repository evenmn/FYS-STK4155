# Import packages
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
from keras.utils import np_utils

import neural_network as nn
from activation import *
from optimization import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import RNN, SimpleRNN, GRU, LSTM
from keras.optimizers import Adam




def load_n_split():
    # Load data set
    X = np.loadtxt('../data/X_40.txt')
    text_file = open('../data/t.txt', 'r')
    t = []
    for line in text_file:
        t.append(line)
        
    # Turn labels into arrays
    lb = LabelEncoder()
    t = np_utils.to_categorical(lb.fit_transform(t))

    # Split into training and test set
    N = len(X)
    N_train = int(N*0.8)
    X_train = X[:N_train]
    t_train = t[:N_train]
    X_val = X[N_train:]
    t_val = t[N_train:]
    return X_train, t_train, X_val, t_val


# Run a deep learning model and get results

def Logistic():
    X_train, t_train, X_val, t_val = load_n_split()

    '''
    model = nn.NeuralNetwork(X_train, t_train, T=100, h=0, eta=0.001, lamb=0.001, f1=logistic, f2=logistic, opt=GD)
    model.solver()                                      # Obtain optimal weights

    y = model.predict(X_train)                     # Recall training phase
    y_val = model.predict(X_val)

    # Need to set the largest value to 1, and the rest to zero

    def Accuracy(y, t):
        y_new = np.zeros(y.shape, dtype=int)
        y_new[np.arange(len(y)), y.argmax(1)] = 1
    
        counter = 0
        for i in range(len(y_new)):
            if np.array_equal(y_new[i], t[i]):
                counter += 1
        return counter/len(y_new)
        
    print(Accuracy(y, t_train))
    print(Accuracy(y_val, t_val))
    '''
    
    num_labels = t_train.shape[1]

    model = Sequential()

    #model.add(Dense(1024, input_shape=(40,)))
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(0.5))

    model.add(Dense(num_labels, input_shape=(40,)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, t_train, batch_size=32, epochs=200, validation_data=(X_val, t_val))
    
    # Examine test dataset
    #X_val = np.loadtxt('../data/X_test_40.txt')
    #y = model.predict(X_val)
    
    #categories = lb.inverse_transform(np.argmax(y, axis=1))
    #for cat in categories:
    #    print(cat)
    

def Neural_network():
    X_train, t_train, X_val, t_val = load_n_split()

    num_labels = t_train.shape[1]

    model = Sequential()

    model.add(Dense(1024, input_shape=(40,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, t_train, batch_size=32, epochs=200, validation_data=(X_val, t_val))
    
    # Examine test dataset
    #X_val = np.loadtxt('../data/X_test_40.txt')
    #y = model.predict(X_val)
    
    #categories = lb.inverse_transform(np.argmax(y, axis=1))
    #for cat in categories:
    #    print(cat)
    
    
def Recurrent():
    X_train, t_train, X_val, t_val = load_n_split()
    
    num_labels = t_train.shape[1]

    model = Sequential()

    model.add(Dense(1024, input_shape=(40,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, t_train, batch_size=32, epochs=200, validation_data=(X_test, t_test))
    
    
if __name__ == '__main__':
    Logistic()
    #Neural_network()
