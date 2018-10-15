import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
import pickle

def read_t(t,root='../data/', reshape=True):
    '''Read data file'''
    
    if isinstance(t, str):
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%s.pkl'%t,'rb'))
    else:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
        
    if reshape:
        return np.unpackbits(data).astype(int).reshape(-1,1600)
    else:
        return data
     


# Load inputs and targets
x = read_t('All')                          # Inputs
t = read_t('All_labels', reshape=False)    # Targets

N = len(x)                  # Number of sets
M = len(x[0])               # Number of elements in each set

# Split into training and validation 
data = np.c_[x, t]                          # Merge x and t before reshuffle
np.random.shuffle(data)                     # Reshuffle

N_training = int((9/10)*N)                  # 90% of the data is used in training
x_training = data[:N_training,:-1]          # Training set
x_test = data[N_training:,:-1]              # Test set
t_training = data[:N_training,-1]           # Training targets
t_test = data[N_training:,-1]               # Targets for validation

'''   
h = np.array([5,5])        #One hidden layers of 3 elements each

W, b = nn.multilayer(x_training, t_training, 1000, h)
net = nn.recall_multilayer(x_test, W, b)
out = np.where(net>0.5,1,0)
'''

h = 5        #One hidden layers of 3 elements each

W, b = nn.nonlinear(x_training, t_training, 1000, h)
net = nn.recall_multilayer(x_test, W, b)
out = np.where(net>0.5,1,0)

print(t_test)
print(out)
