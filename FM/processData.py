#Sailun Xu(sx38)
#Jenny Wang(jw2249)
#Joyce Xu(xx94)

import numpy as np
import re
import pandas as pd
import pickle
import sys

SYS_VERSION = sys.version_info[0]
"""
This is a class for process data.
"""
class ProcessData(object):
	"""
	This method process the data file u.item and return a matrix of which
	the first column is the movie id, the rest are the presence of the 
	movie genre
	"""
	@classmethod
	def parseItem(self,filePath = 'ml-100k/u.item'):
		if(SYS_VERSION == 2):
			f = open(filePath, 'r')
		elif(SYS_VERSION == 3):
			f = open(filePath, 'r', encoding = "ISO-8859-1")
		s = f.read()
		movies = re.split("\n+", s)
		l = len(movies)
		res = np.zeros((l, 19), dtype = np.int)

		for i in range (0, len(movies) - 1):
			ele = movies[i].split('|', 24)
			if(len(ele) == 24): 
				res[i][0] = ele[0]
				for j in range(0, 18):
					res[i][j] = ele[j + 5]
		f.close()
		return res

	@classmethod
	def parseDataBinary(self, path = 'data.txt'):
		temp = open(path, 'rb')
		if(SYS_VERSION == 3):
			A = pickle.load(temp, encoding = 'ISO-8859-1')
		elif(SYS_VERSION == 2):
			A = pickle.load(temp)
		return A

	"""
	This method process the data file u.data and return a matrix of which
	consist of all ratings. For example (i, j) = the rating for item j by
	user i
	"""
	@classmethod
	#returns the sparse matrix from the given data file
	def parseDataRaw(self, path = 'u.data'):
		m = 943 #number of users from u.info
		n = 1683 #number of movies from u.info
		dataSize = 100000 #also from u.info

		A = np.zeros((943,1683))
		data = pd.read_csv('u.data',delimiter = r"\s+",header = None)
		for x in range(0, dataSize - 1):
			m = data.loc[x][0] #user number
			n = data.loc[x][1] #movie number
			rating = data.loc[x][2]
			A[m - 1,n - 1] = rating #offset because of indexing 
		return A

	"""
	This method return the test points coordinates
	"""
	@classmethod
	def testPts(self, path = 'testPoints_100_200.txt'):
		testPoints = open("testPoints_100_200.txt", "r").read().split('\n')[:-1]
		testPoints = [(int(s.split(",")[0]), int(s.split(",")[1])) for s in testPoints]
		return testPoints

# print(ProcessData.parseItem())
# print(ProcessData.testPts())
