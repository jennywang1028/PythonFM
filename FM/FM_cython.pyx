import numpy as np
import scipy as sp
import processData
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse as spa
from scipy import stats
import time
from minimize import minimize
cimport numpy as np

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def predictionForOnec(np.ndarray x, np.ndarray x2, float w0, np.ndarray W, np.ndarray V_T, np.ndarray V2_T):
    cdef float res1,res2,res3
    res1 = np.sum(V_T.dot(x) ** 2)
    res2 = np.sum(V2_T.dot(x2))
    res3 = W.dot(x)

    return w0 + res3 + 0.5 * (res1 - res2)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def predictionc(np.ndarray X, float w0, np.ndarray W, np.ndarray V):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef np.ndarray X2 = X ** 2
    cdef np.ndarray V_T = V.T
    cdef np.ndarray V2_T = (V**2).T

    cdef np.ndarray res = np.zeros(n)
    cdef int i
    for i in range(n):
        res[i] = predictionForOnec(X[i], X2[i], w0, W, V_T, V2_T)

    return res

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def extractc(np.ndarray H, int m, int k):
    cdef double w0 = H[0]
    cdef np.ndarray W = H[1:(m+1)]
    cdef np.ndarray V = (H[(m+1):]).reshape(m,k)
    return w0, W, V

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compressc(float w0, np.ndarray W,\
              np.ndarray V, int m, int k):
    cdef np.ndarray res = np.array([w0])
    res = np.append(res, W)
    res = np.append(res, V.flatten())
    return res    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def gradHVecc(np.ndarray H, np.ndarray X, np.ndarray Y, int m, int k,double reg_weight = 0.05):
    cdef int i,j
    cdef float w0
    cdef np.ndarray W, pred, diff
    cdef np.ndarray V, dLdV, XV,temp, temp2
    cdef int n = X.shape[0]
    cdef float dLdw0
    cdef np.ndarray dLdW,v_ifX,Xv_ifX,res,diffT

    Y = np.array(Y)
    w0, W, V= extractc(H, m, k)

    diff = Y - ( predictionc(X, w0, W, V) ).ravel()

    dLdw0 = - np.sum(diff) * 2 / n
    dLdWi = -2.0/n * (diff.dot(X)).flatten() + 2 * reg_weight * W
    
    dLdV = np.zeros((m, k))
    XV = X.dot(V)
    xT = X.T
    xT2 = xT ** 2
    diffT = diff.reshape(n,1)
    for i in range(m):
        v_ifX = V[i].reshape(k,1) * np.tile(xT2[i],(k, 1)) 
        Xv_ifX = ((xT[i]).reshape(n,-1) * XV).T
        res = (v_ifX - Xv_ifX).dot(diffT)
        dLdV[i] = (res.ravel() * 2.0) / n
    
    return compressc(dLdw0, dLdWi, dLdV, m, k)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def lossHc(np.ndarray H, np.ndarray X, np.ndarray Y, int m, int k,float reg_weight):
    cdef int n = Y.size
    cdef float w0, regularizer, loss
    cdef np.ndarray W, V
    w0, W, V = extractc(H, m, k)
    loss = np.mean((Y - predictionc(X, w0, W, V)) ** 2)
    regularizer = reg_weight * W.dot(W)
    return loss + regularizer