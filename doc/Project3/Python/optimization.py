import numpy as np

def GD(X, t, out, f):
    deltao = (out - t) * f(out, der=True)
    return np.outer(np.transpose(np.insert(X, 0, 1)), deltao)
    
def SGD(X, t, out, f):
    deltao = (out - t) * f(out, der=True)
    return np.outer(np.transpose(np.insert(X, 0, 1)), deltao)
