# Load audio files and extract features
import numpy as np
import librosa
import pandas as pd
import os
import csv

def parser(ID, Class, n_mfcc):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath('../data/'), 'Train', str(ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None
 
   feature = mfccs
   label = Class
   print(ID)
 
   return feature, label

n_mfcc = 40
train = pd.read_csv("../data/train.csv")

X = np.zeros((len(train), n_mfcc))
f = open("../data/t.txt", "w")

for i in range(len(train)):
    feature, label = parser(train.ID[i], train.Class[i], n_mfcc)
    X[i] = feature
    f.write(str(label)+'\n')

np.savetxt("../data/X.txt", X)
f.close()

