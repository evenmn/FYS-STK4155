import os
import pandas as pd
import librosa
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
import csv

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
    else:
        raise NameError("Class %s does not exist" % class_)



train = pd.read_csv("../data/train.csv")

N = 4000
data_train2 = np.zeros((N, 88200))

i = 0
j = 0
while i < N:
    #print(train.ID[j])
    
    if j != 1986 and j != 5312:
        x, sr = librosa.load('../data/Train/' + str(train.ID[j]) + '.wav')
        print(sr)
        if x.shape[0] == 88200:
            data_train2[i] = x
            print(i)
            i += 1
    j += 1

#np.savetxt('../data/data_train.dat', data_train2)

with open('../data/data_train.csv', 'w') as f:
     csv.writer(f, delimiter=' ').writerows(data_train2)

'''
test = pd.read_csv("../data/test.csv")
data_test = []
t_test = []
for i in tqdm(range(test.shape[0])):
    print(i)

    x, sr = librosa.load('../data/Test/' + str(test.ID[i]) + '.wav')

    data_test.append(x)
    t_test.append(test.Class[i])

np.savetxt(np.array(data_test), '../data/data_test.dat')
np.savetxt(np.array(t_test), '../data/t_test.dat')
'''
