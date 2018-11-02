import numpy as np


def MSE(X, J, E):
    '''Mean Square Error'''
    e = X.dot(J) - E
    return e.T.dot(e)/len(E)

    
def R2(X, J, E):
    '''R2 score function'''
    e = X.dot(J) - E
    f = E - np.average(E)
    return 1 - e.T.dot(e)/f.T.dot(f)

    
def Accuracy(y_test, t_test):
    '''Accuracy score'''
    y_test_int = np.where(y_test > 0.5, 1, 0)
    diff = np.sum(np.abs(y_test_int-t_test))

    return 1 - diff/len(t_test)
