import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

train = pd.read_csv("../data/train.csv")

i = np.random.choice(train.index)
x, sr = librosa.load('../data/Train/' + str(train.ID[i]) + '.wav')

t = np.linspace(0,int(len(x)/sr),len(x))

# Plot
FS = 14
#plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
plt.plot(t,x)
plt.title("%s"%train.Class[i], fontsize=FS)
plt.xlabel("Time [s]", fontsize=FS)
plt.ylabel("Amplitude", fontsize=FS)


# Frequency domain
FFT = np.fft.fft(x)
print(len(FFT))
plt.subplot(2,1,2)
plt.plot(FFT)
plt.xlabel("Frequency [Hz]", fontsize=FS)
plt.ylabel("Amplitude", fontsize=FS)
plt.show()
