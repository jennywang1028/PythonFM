from BatchGD import *

test_x = np.array([1,2,3,0,4])
test_w0 = 3
test_W = np.array([2,1,0,4,3])
test_V = np.array([[1,2,3],
                   [1,3,2],
                   [4,1,2],
                   [2,1,0],
                   [0,1,2]])
assert(predictionForOneOld(test_x, test_w0, test_W, test_V) == 295)
assert(predictionForOne(test_x, (test_x ** 2), test_w0, test_W, test_V.T, (test_V ** 2).T) == 295)