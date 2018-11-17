import os
import pandas as pd
import librosa
import glob 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../data/train_fuSp8nd.csv")

#data, sampling_rate = librosa.load('../data/2022.wav')

print(data)
stop

plt.figure(figsize=(12, 4))
plt.plot(data)
plt.show()
