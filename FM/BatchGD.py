'''
This implements Factorization Machine by Rendle.
Unlike other implementation, this implementation uses only python features. 
'''
import numpy as np
import scipy as sp
import os
# import processData
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse as spa
from scipy import stats
import time
from time import clock
import heapq
import operator
from minimize import minimize

FACTOR = 1

W0_ADDR = "../Saved/w0.pkl"
W_ADDR = "../Saved/w.pkl"
V_ADDR = "../Saved/V.pkl"

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

'''
Computes loss for given y, prediction, W, and regularizer weight. 
'''
def L(Y, prediction, W, reg_weight = 0.01,loss = "squared"):
    if(loss == "squared"):
        n = Y.size
        loss = np.sum((Y - prediction) ** 2)/n
        regularizer = reg_weight * W.dot(W)
        return loss + regularizer

'''
Computes the correctness for the given y and prediction.
Only 
'''
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
    # if FACTOR == 0:
    #     FACTOR = 1
    times = int(n / FACTOR)
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

def load_data_pred(data_input):
    data = []
    y = []
    users = set()
    items = set()
    for line in data_input: 
        (user,movieid,rating,ts) = line.split('\t')
        data.append({"user_id" : str(user), "movie_id" : str(movieid)})
        y.append(float(rating))
        users.add(user)
        items.add(movieid)
    return (data, np.array(y), users, items)


def gen_test(users, movies = 300):
    test = []
    for u in users:
        for m in range(1, 1000):
            test.append({"user_id" : str(u), "movie_id" : str(m)})
    return test


def predict(input_data, train_addr = "ua.base", test_addr = "ua.test", nb_epochs = 4, k = 10, n = 10):
    """
    input_data: should follow the format as in ua.base
    train_addr: training data address
    test_addr: testing adta address, not used if called in front-end
    nb_epochs: number of epochs, default 4
    n: return the top nth highest movie id, default 10
    """
    pre_trained = os.path.exists(W0_ADDR) and os.path.exists(W_ADDR) and os.path.exists(V_ADDR)
    (train_data, y_train, train_users, train_items) = loadData(train_addr)
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    (input_train, input_y, input_users, input_items) = load_data_pred(input_data)
    I_train = v.transform(input_train)
    X_train = np.array(X_train.todense())
    I_train = np.array(I_train.todense())
    y_train = np.array(y_train)
    input_y = np.array(input_y)

    if pre_trained:
        w0 = np.load(W0_ADDR)
        W = np.load(W_ADDR)
        V = np.load(V_ADDR)
        w0, W, V = runMinVec(I_train,input_y,w0, W, V, k)

        test = v.transform(gen_test(input_users))
        test = np.array(test.todense())

        pred = prediction(test, w0, W, V)

    else:
        n, m = X_train.shape
        w0 = np.random.rand()
        W  = np.random.rand(m)
        V  = np.random.rand(m, k)

        for i in range(nb_epochs):
            w0, W, V = runMinVec(X_train,y_train,w0, W, V, k)

        w0.dump(W0_ADDR)
        W.dump(W_ADDR)
        V.dump(V_ADDR)

        pred = prediction(X_test, w0, W, V)
    
    res = zip(*heapq.nlargest(n, enumerate(pred), key=operator.itemgetter(1)))[0]
    return res

def make_batches(data, batch_size):
    batches = []
    batch = []
    for rate in data: 
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
        batch.append(rate)
    if batch:
        batches.append(batch)

    return np.array(batches)




if __name__ == '__main__':

    (train_data, y_train, train_users, train_items) = loadData("ua.base")
    (test_data, y_test, test_users, test_items) = loadData("ua.test")
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)
    y_train = np.array(y_train)
    y_test = y_test

    X_train = np.array(X_train.todense())
    X_test = np.array(X_test.todense())
    k = 10
    nb_epochs = 4
    n, m = X_train.shape
    w0 = np.load(W0_ADDR)
    W  = np.load(W_ADDR)
    V  = np.load(V_ADDR)
    epo_losses = []
    correctness = []
    rmse = []
    avg_dist = []
    
    for i in range(nb_epochs):
        print("running epoch{}".format(i))
        w0, W, V = runMinVec(X_train,y_train,w0, W, V, k)
        pred = prediction(X_test, w0, W, V)
        print("loss is:{}".format(L(y_test, pred,W)))
        epo_losses.append(L(y_test, pred, W))
    #     print("y_test: ", y_test, "pred: ", pred)
    #     print("correctness is:{}".format(crtness(y_test, pred)))
        correctness.append(crtness(y_test, pred))
        rmse.append(RMSE(y_test, pred))
        avg_dist.append(avgdist(y_test, pred))
        crt = crtness(y_test, pred)
        avgd = avgdist(y_test, pred)


        print (("correctness is:{}. \n"
                        "RMSE is: {}. \n"
                        "Average distance is: {}. ").format(crt,RMSE(y_test,pred),avgd))

    # w0.dump(W0_ADDR)
    # W.dump(W_ADDR)
    # V.dump(V_ADDR)

    # w0 = np.load(W0_ADDR)
    # W = np.load(W_ADDR)
    # V = np.load(V_ADDR)
    # (train_data, y_train, train_users, train_items) = loadData("ua.base")
    # v = DictVectorizer()
    # X_train = v.fit_transform(train_data)
    # (test_data, y_test, test_users, test_items) = loadData("ua.test")
    # X_test = v.transform(test_data)
    # X_test = np.array(X_test.todense())
    # preds = prediction(X_test, w0, W, V)


    # crt = crtness(y_test, preds)
    # rmse = RMSE(y_test,preds)
    # avgd = avgdist(y_test, preds)


    # print ("correctness is:{}. \n"
    #         "RMSE is: {}. \n"
    #         "Average distance is: {}. ").format(crt,rmse,avgd)
