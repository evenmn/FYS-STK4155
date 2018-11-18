import os
import pandas as pd
import librosa
import glob 
import pandas as pd
#from tqdm import tqdm
import wavio


train = pd.read_csv("../data/train.csv")
data_train = []
t_train = []
for i in range(train.shape[0]):
    print(i)

    audio_name = train.ID[i]
    path = os.path.join('../data/Train', str(audio_name) + '.wav')

    x, sr = librosa.load('../data/Train/' + str(train.ID[i]) + '.wav')
    data_train.append(x)
    t_train.append(sr)

np.savetxt(np.array(data_train), '../data/data_train.dat')
np.savetxt(np.array(t_train), '../data/t_train.dat')


test = pd.read_csv("../data/test.csv")
data_test = []
t_test = []
for i in range(test.shape[0]):
    print(i)

    x, sr = librosa.load('../data/Test/' + str(test.ID[i]) + '.wav')
    #sr, x = wavfile.read('../data/Test/' + str(test.ID[i]) + '.wav')
    #sr, x = wavio.read('../data/Test/' + str(test.ID[i]) + '.wav')

    data_test.append(x)
    t_test.append(test.Class[i])

np.savetxt(np.array(data_test), '../data/data_test.dat')
np.savetxt(np.array(t_test), '../data/t_test.dat')
