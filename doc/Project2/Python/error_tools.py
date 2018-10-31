import numpy as np

'''
def MSE(J, J_):
    Mean Square Error
    e = J - J_
    return e.T.dot(e)/len(J)
    
    
def R2(x, y, z, beta):
    R2 score function
    e = z - polyval(x, y, beta)
    f = y - np.average(y)
    
    return 1 - e.T.dot(e)/f.T.dot(f)
'''
    
def Accuracy(y_test, t_test):
    '''Accuracy score'''
    y_test_int = np.where(y_test > 0.5, 1, 0)
    diff = np.sum(np.abs(y_test_int-t_test))

    return 1 - diff/len(t_test)
