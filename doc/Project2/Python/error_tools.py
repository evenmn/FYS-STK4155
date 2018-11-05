import numpy as np

    
def MSE(E_tilde, E):
    '''Mean Square Error'''
    e = E_tilde - E
    return e.T.dot(e)/len(E)
    
    
def R2(E_tilde, E):
    '''R2 score function'''
    e = E_tilde - E
    f = E - np.average(E)
    return 1 - e.T.dot(e)/f.T.dot(f)
    
    
def MSE_linreg(X, J, E):
    '''Mean Square Error in linear regression'''
    E_tilde = X.dot(J)
    return MSE(E_tilde, E)
    
    
def R2_linreg(X, J, E):
    '''R2 score function in linear regression'''
    E_tilde = X.dot(J)
    return R2(E_tilde, E)

    
def Accuracy(y_test, t_test):
    '''Accuracy score'''
    y_test_int = np.where(y_test > 0.5, 1, 0)
    diff = np.sum(np.abs(y_test_int-t_test))

    return 1 - diff/len(t_test)
