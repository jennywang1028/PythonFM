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


def RMSE(y, pred):
    diff = (y - pred) ** 2
    res = np.sqrt( np.mean(diff) )
    return res

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

def crtness(Y, prediction):
    n = Y.size
    prediction = np.round(prediction)
    prediction = prediction.astype(int)
    return (n - np.count_nonzero(Y - prediction)) / float(n)

def avgdist(Y, prediction):
    n = Y.size
    return np.mean(np.abs(Y - prediction))

if __name__ == '__main__':
	(train_data, y_train, train_users, train_items) = loadData("ua.base")
	(test_data, y_test, test_users, test_items) = loadData("ua.test")
	
	mode,_ = stats.mode(y_train)
	mode_pred = np.full(len(y_test), mode , dtype = int)
	mode_crt = crtness(y_test, mode_pred)
	mode_rmse = RMSE(y_test, mode_pred)
	mode_avgdist = avgdist(y_test, mode_pred)

	print (("correctness is:{}. \n"
                "RMSE is: {}. \n"
                "Average distance is: {}. ").format(mode_crt,mode_rmse,mode_avgdist))
