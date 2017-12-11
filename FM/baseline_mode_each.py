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

def predict(train_data, test_data):
	dic = {}
	y_train = []
	with open(train_data, 'r') as f: 
		for line in f:
			w = line.split('\t')
			if w[1] in dic:
				dic[w[1]].append(int(w[2]))
			else:
				dic[w[1]] = [int(w[2])]
			y_train.append(int(w[2]))

	m_Y , _ = stats.mode(y_train) 
	pred = []
	with open (test_data, 'r') as f: 
		for line in f: 
			w = line.split('\t')
			if w[1] in dic: 
				l = dic[w[1]]
				m, _ = stats.mode(dic[w[1]])
				pred.append(int(m))
			else:
				pred.append(int(m_Y))

	return pred

if __name__ == '__main__':
	preds = predict("../ml-100k/ua.base", "../ml-100k/ua.test")
	(test_data, y_test, test_users, test_items) = loadData("ua.test")

	crt = crtness(y_test, preds)
	rmse = RMSE(y_test,preds)
	avgd = avgdist(y_test, preds)


	print( ("correctness is:{}. \n"
				"RMSE is: {}. \n"
				"Average distance is: {}. ").format(crt,rmse,avgd))





