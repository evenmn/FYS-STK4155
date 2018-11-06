import numpy as np

K = 10

X = np.array([[0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9]])
E = np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9])

#print(X)



Xmat = np.reshape(X, (K, int(len(X)/K), len(X[0])))
Emat = np.reshape(E, (K, int(len(X)/K)))

for i in range(K):
    Xnew = np.delete(Xmat, i, 0)
    Enew = np.delete(Emat, i, 0)

    print(Xnew)
    
    X_train = np.reshape(Xnew, (len(Xnew)*len(Enew[0]), len(X[0])))
    E_train = np.reshape(Enew, (len(Xnew)*len(Enew[0])))
    
    #print(X)
    print(X_train)
