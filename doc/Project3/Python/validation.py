import numpy as np
import librosa
import pandas as pd
import IPython.display as ipd

test = pd.read_csv('../data/test.csv')

i = np.random.choice(test.index)

audio_name = test.ID[i]
ipd.Audio('../data/Test/' + str(audio_name) + '.wav')
#path = os.path.join('../data/Test', str(audio_name) + '.wav')

#x, sr = librosa.load('../data/Test/' + str(train.ID[i]) + '.wav')


