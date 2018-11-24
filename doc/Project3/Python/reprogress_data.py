import librosa
import pandas as pd
import numpy as np

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

np.savetxt('../data/data_train.dat', data_train)
np.savetxt('../data/t_train.dat', t_train)

#with open('../data/data_train.csv', 'w') as f:
#     csv.writer(f, delimiter=' ').writerows(data_train2)
