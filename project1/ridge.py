
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE

    # d = len(xTr)
    # n = len(xTr[0])

    sqDiff = np.dot(np.transpose(w), xTr) - yTr
    loss = np.dot(sqDiff, np.transpose(sqDiff)) + lambdaa * np.dot(np.transpose(w), w)
    gradient = 2 * (np.dot(np.dot(xTr, np.transpose(xTr)), w) - np.dot(xTr, np.transpose(yTr))
        + lambdaa * w)
        
    #print(loss)
    return loss,gradient