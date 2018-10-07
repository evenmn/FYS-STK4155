import numpy as np
from regression_2D import *
from error_tools import *

def bootstrap(data, K=1000):
    '''Bootstrap resampling
    Credits to Morten Hjorth-Jensen for implementation'''
    
    dataVec = np.zeros(K)
    for k in range(K):
        dataVec[k] = np.average(np.random.choice(data, len(data)))
        
    Avg = np.average(dataVec)
    Var = np.var(dataVec)
    Std = np.std(dataVec)
    
    return Avg, Var, Std
    
    
    
def k_fold(x, y, z, K=8):
    '''K-fold validation resampling'''
    
    xMat = np.reshape(x, (K, int(len(x)/K)))
    yMat = np.reshape(y, (K, int(len(y)/K)))
    zMat = np.reshape(z, (K, int(len(z)/K)))
    
    MSE_train = 0
    MSE_test = 0
    
    for i in range(K):
        xMatNew = np.delete(xMat, i, 0)
        yMatNew = np.delete(yMat, i, 0)
        zMatNew = np.delete(zMat, i, 0)
        
        xVecNew = xMatNew.flatten()
        yVecNew = yMatNew.flatten()
        zVecNew = zMatNew.flatten()
        
        order5 = Reg_2D(xVecNew, yVecNew, zVecNew, Px=5, Py=5)
        beta_train = order5.ols()
        
        MSE_train += MSE(xVecNew, yVecNew, zVecNew, beta_train)
        MSE_test += MSE(xMat[i], yMat[i], zMat[i], beta_train)

    return MSE_train/K, MSE_test/K
        
    

def blocking(data):
    '''Blocking method
    Credits to Marius Jonsson for implementation'''
    
    # preliminaries
    n = len(data)
    d = int(np.floor(np.log2(n)))
    s = gamma = np.zeros(d)
    mu = np.mean(data)

    # estimate the auto-covariance and variances 
    # for each blocking transformation
    for i in np.arange(0,d):
        n = len(data)
        # estimate autocovariance of data
        gamma[i] = (n)**(-1)*np.sum((data[0:(n-1)]-mu)*(data[1:n]-mu) )
        # estimate variance of data
        s[i] = np.var(data)
        # perform blocking transformation
        data = 0.5*(data[0::2] + data[1::2])
   
    # generate the test observator M_k from the theorem
    M = (np.cumsum(((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = np.array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894, \
               18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967, \
               27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306, \
               36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820, \
               44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
        
    Avg = mu
    Var = s[k]/2**(d-k)
    Std = Var**.5
    
    return Avg, Var, Std
    

if __name__ == '__main__':

    '''Simple running example'''
    from numpy.random import normal

    x = normal(0,1,2**7)
    
    print(bootstrap(x))
    print(k_fold(x))
    print(blocking(x))
