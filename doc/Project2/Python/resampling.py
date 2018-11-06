import numpy as np
from error_tools import *
from sklearn import linear_model
from ising_data import generate_J
import neural_network as nn

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

        
def k_fold(X, E, eta=1e-3, K=10):
    '''K-fold validation resampling based on neural netowrk'''
    
    MSE_train = 0
    MSE_test = 0
    R2_train = 0
    R2_test = 0
    
    Xmat = np.reshape(X, (K, int(len(X)/K), len(X[0])))
    Emat = np.reshape(E, (K, int(len(X)/K)))
    
    for i in range(K):
        Xnew = np.delete(Xmat, i, 0)
        Enew = np.delete(Emat, i, 0)
        
        X_train = np.reshape(Xnew, (len(Xnew)*len(Enew[0]), len(X[0])))
        E_train = np.reshape(Enew, (len(Xnew)*len(Enew[0])))
        
        W = nn.linear(X_train, E_train, 50)
        E_train_tilde = nn.recall_linear(X_train, W)
        E_test_tilde = nn.recall_linear(Xmat[i], W)
        
        MSE_train += MSE(E_train_tilde, E_train)
        MSE_test += MSE(E_test_tilde, Emat[i])
        
        R2_train += R2(E_train_tilde, E_train)
        R2_test += R2(E_test_tilde, Emat[i])

    return MSE_train/K, MSE_test/K, R2_train/K, R2_test/K
        
        
def k_fold_linreg(X, E, λ=1e-4, K=10, method='J_ols'):
    '''K-fold validation resampling based on linear algebra'''
    
    MSE_train = 0
    MSE_test = 0
    R2_train = 0
    R2_test = 0
    
    Xmat = np.reshape(X, (K, int(len(X)/K), len(X[0])))
    Emat = np.reshape(E, (K, int(len(X)/K)))
    
    for i in range(K):
        Xnew = np.delete(Xmat, i, 0)
        Enew = np.delete(Emat, i, 0)
        
        X_train = np.reshape(Xnew, (len(Xnew)*len(Enew[0]), len(X[0])))
        E_train = np.reshape(Enew, (len(Xnew)*len(Enew[0])))
        
        
        if method == 'J_ols':
            ols=linear_model.LinearRegression()
            ols.fit(X_train, E_train)
            J = ols.coef_

        elif method == 'J_ridge':
            ridge=linear_model.Ridge()
            ridge.set_params(alpha=λ)
            ridge.fit(X_train, E_train)
            J = ridge.coef_

        elif method == 'J_lasso':
            lasso=linear_model.Lasso()
            lasso.set_params(alpha=λ)
            lasso.fit(X_train, E_train)
            J = lasso.coef_
        
        else:
            raise NameError("No method named ", method)
        
        MSE_train += MSE_linreg(X_train, J, E_train)
        MSE_test += MSE_linreg(Xmat[i], J, Emat[i])
        
        R2_train += R2_linreg(X_train, J, E_train)
        R2_test += R2_linreg(Xmat[i], J, Emat[i])

    return MSE_train/K, MSE_test/K, R2_train/K, R2_test/K
    

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
