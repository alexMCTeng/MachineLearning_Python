from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE

    ywx = np.multiply(yTr, w.T.dot(xTr))
    ywx = ywx.ravel()
    #print(ywx)
    loss_tmp = 1 - ywx
    loss_boolean = loss_tmp > 0
    #print(tmp2)
    loss_tmp[np.where(ywx >= 1)] = 0
    loss = np.sum(np.multiply(loss_tmp , loss_boolean)) + w.T.dot(w) * lambdaa
    #loss = np.sum(tmp) + w.T.dot(w) * lambdaa
    #print(loss)

    #print(ywx)

    #xTr[:,np.where(ywx >= 1)] = 0
    #yTr[np.where(ywx >= 1)] = 0
    grad = - np.multiply(yTr, xTr) * loss_boolean
    gradient = np.sum(grad, axis=1).reshape(w.shape) + 2 * lambdaa * w
    return loss,gradient
