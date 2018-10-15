import numpy as np
from regression_2D import polyval

def MSE(J, J_):
    '''Mean Square Error'''
    e = J - J_
    return e.T.dot(e)/len(J)
    
    
def R2(x, y, z, beta):
    '''R2 score function'''
    e = z - polyval(x, y, beta)
    f = y - np.average(y)
    
    return 1 - e.T.dot(e)/f.T.dot(f)
    
