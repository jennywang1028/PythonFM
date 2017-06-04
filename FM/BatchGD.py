import numpy as np
import scipy as sp
import processData
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse as spa
from scipy import stats
import time
from minimize import minimize

def predictionForOne(x, x2, w0, W, V_T, V2_T):
    # m = x.size
    # x = x.reshape(m, -1)
    # print(W.shape)
    res1 = V_T.dot(x)
    res1 = np.sum(res1 ** 2)
    res2 = V2_T.dot(x2)
    res2 = np.sum(res2)
    return w0 + W.dot(x.flatten()) + 0.5 * (res1 - res2)

'''
Generates prediction y for given X, w0, W, V following the definition of factorization machine
'''
def prediction(X,w0,W,V):
    res1 = X.dot(V)
    res1 = 0.5 * np.sum(res1**2, axis = 1)
    res2 = (X**2).dot(V**2)
    res2 = 0.5 * np.sum(res2, axis = 1)
    t = X.dot(W)
    res1 -= res2
    res1 += w0
    res1 += t
    return res1


def L(Y, prediction, W, reg_weight = 0.01,loss = "squared"):
    if(loss == "squared"):
        n = Y.size
        loss = np.sum((Y - prediction) ** 2)/n
        regularizer = reg_weight * W.dot(W)
        return loss + regularizer

def crtness(Y, prediction):
    n = Y.size
    prediction = np.round(prediction)
    prediction = prediction.astype(int)
    return (n - np.count_nonzero(Y - prediction)) / float(n)

def softcrtness(Y, prediction):
    n = Y.size
    prediction = np.round(prediction)
    prediction = prediction.astype(int)
    soft_crt = 0
    for i in range(len(Y)):
        if(abs(Y[i] - prediction[i]) <= 1):
            soft_crt += 1
    return soft_crt/float(n)

def avgdist(Y, prediction):
    n = Y.size
    return np.mean(np.abs(Y - prediction))

def evaluateGradient(X, w0, W, V, Y, k, reg_weight = 0.05):
    Y = np.array(Y)
    n,m = X.shape
    pred = prediction(X, w0, W, V)
    pred = np.array(pred).flatten()
    regularizer = reg_weight * W.dot(W)
    diff = Y - pred

    dLdw0 = - np.sum(diff) * 2 / n
    dLdWi = -2.0/n * (diff.dot(X)).flatten() + 2 * reg_weight * W
    
    start_time = time.time() # evaluate running time

    dLdV = np.zeros((m, k))
    XV = X.dot(V) 
    for i in range(m):
        temp = np.array(X[:,i]).reshape(-1)
        temp2 = temp ** 2
        for f in range(k):
            v_ifX = V[i][f] * temp2
            Xv_ifX = temp * (XV[:,f].reshape(-1))
            dLdV[i][f] = diff.dot(v_ifX - Xv_ifX)
    dLdV = dLdV * 2.0 / n
    print("--- %s seconds ---" % (time.time() - start_time))
    return dLdw0, dLdWi, dLdV


def evaluateGradientVec(X, w0, W, V, Y, k, reg_weight = 0.05):
    Y = np.array(Y)
    n,m = X.shape
    diff = Y - (np.asarray(prediction(X, w0, W, V))).ravel()

    dLdw0 = - np.sum(diff) * 2 / n
    dLdWi = -2.0/n * (diff.dot(X)).ravel() + 2 * reg_weight * W
    
    dLdV = np.zeros((m, k))
    XV = X.dot(V)
    xT = X.T
    xT2 = xT ** 2
    diffT = diff.reshape(n,1)
    for i in range(m):
        # v_ifX = V[i].reshape(k,-1) * xT2[i]
        # v_ifX = (np.expand_dims(V[i], axis = 0).T).dot(np.expand_dims(xT2[i], axis = 0))
        # v_ifX = V[i].reshape(k,1) * np.tile(xT2[i],(k, 1)) # do matrix mul matrix
        # v_ifX = V[i].reshape(-1,1).dot((xT2[i]).reshape(1, -1))
        v_ifX = (V[i, np.newaxis].T).dot(xT2[i, np.newaxis])
        Xv_ifX = (xT[i][:,np.newaxis] * XV).T
        res = (v_ifX - Xv_ifX).dot(diffT)
        dLdV[i] = res.ravel()
    dLdV *= 2/n
    return dLdw0, dLdWi, dLdV

'''
---------The following are modified functions for using minimize.py--------
'''

'''
Compress w0, W, V into a huge array
'''
def compress(w0, W, V, m, k):
    res = np.array([w0])
    res = np.append(res, W)
    res = np.append(res, V.flatten())
    return res    
'''
Extract w0, W, V from H into separate data
'''   
def extract(H, m, k):
    w0 = H[0]
    W = H[1:(m+1)]
    V = (H[(m+1):]).reshape(m,k)
    return w0, W, V

'''
Loss function for H, the only difference between this and L is 
there is a step to extract 
'''
def lossH(H, X, Y, m, k, reg_weight):
    n = Y.size
    w0, W, V = extract(H, m, k)
    pred = prediction(X, w0, W, V)
    loss = np.sum((Y - pred) ** 2)/n
    regularizer = reg_weight * W.dot(W)
    return loss + regularizer

def gradH(H, X, Y, m, k, reg_weight):
    Y = np.array(Y)
    n,m = X.shape
    w0, W, V = extract(H, m, k)
#     start_time = time.time() # evaluate running time
    pred = prediction(X, w0, W, V)
#     print("--- %s seconds predicting...---" % (time.time() - start_time))
    pred = np.array(pred).flatten()
    regularizer = reg_weight * W.dot(W)
    diff = Y - pred

    dLdw0 = - np.sum(diff) * 2 / n
    dLdWi = -2.0/n * (diff.dot(X)).flatten() + 2 * reg_weight * W
    


    dLdV = np.zeros((m, k))
    XV = X.dot(V) 
    for i in range(m):
        temp = np.array(X[:,i]).reshape(-1)
        temp2 = temp ** 2
        for f in range(k):
            v_ifX = V[i][f] * temp2
            Xv_ifX = temp * (XV[:,f].reshape(-1))
            dLdV[i][f] = diff.dot(v_ifX - Xv_ifX)
    dLdV = dLdV * 2.0 / n
    return compress(dLdw0, dLdWi, dLdV, m, k)


# def gradHVecOLD(H, X, Y, m, k, reg_weight):
#     Y = np.array(Y)
#     n,m = X.shape
#     w0, W, V = extract(H, m, k)
#     diff = Y - (np.asarray(prediction(X, w0, W, V))).ravel()

#     dLdw0 = - np.sum(diff) * 2 / n
#     dLdWi = -2.0/n * (diff.dot(X)).ravel() + 2 * reg_weight * W

#     # dLdV = np.empty((m,k))
#     # XV = X.dot(V)
#     # xT2 = xT ** 2
#     # v_ifX = np.empty((k,n))
#     # Xv_ifX = np.empty((k,n))




#     XVT = X.dot(V).T
#     xT = X.T
#     diffT = diff[:, np.newaxis]
#     x2 = X ** 2

#     # print(x2.shape)
#     # print(V.shape)


    # v_ifXDiff = np.einsum('ij, jk -> jki', x2, V)
    # Xv_ifX = np.einsum('ij, kj -> ikj', xT, XVT)
    # v_ifXDiff -= Xv_ifX
    # dLdV = np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT).reshape(m,k)

#     # v_ifX = np.einsum('ij, jk -> jki', x2, V) - np.einsum('ij, kj -> ikj', xT, XVT)
#     # v_ifXDiff -= np.einsum('ij, kj -> ikj', xT, XVT)

#     # Xv_ifX = np.einsum('ij, kj -> ikj', xT, XVT)


#     # dLdV = np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT).reshape(m,k)

#     dLdV = np.array([])
#     times = 10
#     for i in range(times):
#         if i < times - 1:
#             v_ifXDiff = np.einsum('ij, jk -> jki', x2[i*int(n/times):(i+1)*int(n/times),:], V[:,i*int(k/times):(i+1)*int(k/times)])
#             Xv_ifX = np.einsum('ij, kj -> ikj', xT[:,i*int(n/times):(i+1)*int(n/times)], XVT[i*int(k/times):(i+1)*int(k/times),i*int(n/times):(i+1)*int(n/times)])
#             v_ifXDiff -= Xv_ifX
#             dLdV = np.append(dLdV, np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT[i*int(n/times):(i+1)*int(n/times)]))
# #             dLdV.append(np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT[i*int(n/times):(i+1)*int(n/times)]))
#         else:
#             v_ifXDiff = np.einsum('ij, jk -> jki', x2[i*int(n/times):,:], V[:,i*int(k/times):])
#             Xv_ifX = np.einsum('ij, kj -> ikj', xT[:,i*int(n/times):], XVT[i*int(k/times):,i*int(n/times):])
#             v_ifXDiff -= Xv_ifX
#             dLdV = np.append(dLdV, np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT[i*int(n/times):]))
# #             dLdV.append(np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT[i*int(n/times):]))

#     dLdV = dLdV.reshape(m,k)


#     dLdV *= 2/n
    # for i in range(m):
    #     # v_ifX = V[i].reshape(k,-1) * xT2[i]
    #     # v_ifX = (np.expand_dims(V[i], axis = 0).T).dot(np.expand_dims(xT2[i], axis = 0))
    #     # v_ifX = V[i].reshape(k,1) * np.tile(xT2[i],(k, 1)) # do matrix mul matrix
    #     v_ifX = V[i].reshape(-1,1).dot((xT2[i]).reshape(1, -1))
    #     # v_ifX = np.multiply.outer(V[i], xT2[i]) # V: m x k X: n x m 

    #     # v_ifX = (V[i, np.newaxis].T).dot(xT2[i, np.newaxis])
    #     # np.dot(V[i].reshape(-1,1), xT2[i].reshape(1,-1), v_ifX)

    #     # np.dot((V[i, np.newaxis].T),xT2[i, np.newaxis],  v_ifX)
        
    #     Xv_ifX = ((xT[i]).reshape(n,-1) * XV).T # XV: n x k x: n x m
    #     # Xv_ifX = np.transpose(((xT[i]).reshape(n,-1) * XV))

    #     # Xv_ifX = (xT[i][:,np.newaxis] * XV).T
    #     # np.multiply(xT[i], XVT, Xv_ifX)
    #     # Xv_ifX = xT[i] * XVT 

    #     res = (v_ifX - Xv_ifX).dot(diffT)
    #     dLdV[i] = res.ravel()
#     return compress(dLdw0, dLdWi, dLdV, m, k)



def gradHVec(H, X, Y, m, k, reg_weight):
    Y = np.array(Y)
    n,m = X.shape
    w0, W, V = extract(H, m, k)
    diff = Y - (np.asarray(prediction(X, w0, W, V))).ravel()
    dLdw0 = - np.sum(diff) * 2 / n
    dLdWi = -2.0/n * (diff.dot(X)).ravel() + 2 * reg_weight * W
    XVT = X.dot(V).T
    xT = X.T
    diffT = diff[:, np.newaxis]
    x2 = X ** 2
    dLdV = np.array([])
    factor = 50
    times = int(n / factor)
    if times == 0:
        times = 1
    for i in range(times):
        if i < times - 1:
            v_ifXDiff = np.einsum('ij, jk -> jki', x2[i*int(n/times):(i+1)*int(n/times),:], V[:,i*int(k/times):(i+1)*int(k/times)])
            Xv_ifX = np.einsum('ij, kj -> ikj', xT[:,i*int(n/times):(i+1)*int(n/times)], XVT[i*int(k/times):(i+1)*int(k/times),i*int(n/times):(i+1)*int(n/times)])
            v_ifXDiff -= Xv_ifX
            dLdV = np.append(dLdV, np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT[i*int(n/times):(i+1)*int(n/times)]))
        else:
            v_ifXDiff = np.einsum('ij, jk -> jki', x2[i*int(n/times):,:], V[:,i*int(k/times):])
            Xv_ifX = np.einsum('ij, kj -> ikj', xT[:,i*int(n/times):], XVT[i*int(k/times):,i*int(n/times):])
            v_ifXDiff -= Xv_ifX
            dLdV = np.append(dLdV, np.einsum('ijk , kl ->ijl', v_ifXDiff, diffT[i*int(n/times):]))

    dLdV = dLdV.reshape(m,k)
    dLdV *= 2/n

    return compress(dLdw0, dLdWi, dLdV, m, k)

'''
Wrapper for running minimize.py
'''
def runMin(X,Y, w0, W, V, k, reg_weight = 0.05):
    n,m = X.shape
    H = compress(w0, W, V, m, k)
    [newH, fx, i] = minimize(H, lossH, gradH, (X, Y, m, k, reg_weight),maxnumlinesearch=8, verbose=False)
    H = newH
    
    return extract(H, m, k)

def runMinVec(X,Y, w0, W, V, k, reg_weight = 0.05):
    n,m = X.shape
    H = compress(w0, W, V, m, k)
    [newH, fx, i] = minimize(H, lossH, gradHVec, (X, Y, m, k, reg_weight),maxnumlinesearch=8, verbose=False)
    H = newH
    
    return extract(H, m, k)

def loadData(filename,path="../ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

def predFix(datas, test, fix):
    if fix == "row":
        res = np.zeros( (datas.shape[0], test.shape[1]) )
        for i in range(datas.shape[0]):
            sum = 0.0
            count = 0.0
            for j in range(datas.shape[1]):
                if(datas[i][j] != 0):
                    sum += datas[i][j]
                    count += 1
            res[i, :].fill(sum / count)
    else:
        res = np.zeros( (datas.shape[0], test.shape[1]))
        for i in range (test.shape[1]):
            sum = 0.0
            count = 0.0
            for j in range(datas.shape[0]):
                if(datas[j][i] != 0):
                    sum += datas[j][i]
                    count += 1
            if count == 0:
                res[:,i].fill(0)
            else:
                res[:,i].fill (sum / count)
    return res

def extractPred(preds, test, length):
    t = 0
    res = np.zeros(length)
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if(test[i][j] != 0):
                res[t] = preds[i][j]
                t += 1
    return res

def RMSE(y, pred):
    diff = (y - pred) ** 2
    res = np.sqrt( np.mean(diff) )
    return res
