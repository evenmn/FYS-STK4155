import numpy as np
import pickle

def read_pkl(t, path='../data/', reshape=True):
    '''Read data from .pkl file'''
    
    if isinstance(t, str):
        data = pickle.load(open(path+'Ising2DFM_reSample_L40_T=%s.pkl'%t,'rb'))
    else:
        data = pickle.load(open(path+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
        
    if reshape:
        return np.unpackbits(data).astype(int).reshape(-1,1600)
    else:
        return data
     
     
def write_file(data, extension='.csv', path='../data/'):
    '''Write file'''
    np.savetxt(path+'Ising2DFM_reSample_L40tcexcluded_shuffled%s'%extension, data, delimiter=',')


def ignore_tc():
    '''All data but the critical'''
    # Load inputs and targets
    x = read_pkl('All')                          # Inputs
    t = read_pkl('All_labels', reshape=False)    # Targets

    data = np.int_(np.c_[x, t])                  # Merge x and t
    data = np.delete(data, np.arange(70000, 100000), axis=0)    # Remove data around tc

    np.random.shuffle(data)
    return data
    
    
def all_data():
    '''All data'''
    # Load inputs and targets
    x = read_pkl('All')                          # Inputs
    t = read_pkl('All_labels', reshape=False)    # Targets

    data = np.int_(np.c_[x, t])                  # Merge x and t
    
    np.random.shuffle(data)
    return data
    
    
def tc():
    '''Critical data'''
    # Load inputs and targets
    x = read_pkl('All')                          # Inputs
    t = read_pkl('All_labels', reshape=False)    # Targets

    data = np.int_(np.c_[x, t])                  # Merge x and t
    data = data[70000:100000]

    np.random.shuffle(data)
    return data
