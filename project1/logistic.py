import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE

    # end with [1, n] for the loss

    loss = 0
    gradient = 0
        #print(np.exp(-np.dot(yTr[0, i], np.dot(w.T, xTr[:, i]))))
        

    ewxy= np.exp(-np.multiply(w.T.dot(xTr), yTr))
    loss = np.sum(np.log(1 + ewxy), axis=1)
    #print(loss.shape)
    
    ewxy= np.exp(np.multiply(yTr , np.dot(w.T, xTr)))
    gradient = -np.sum(np.multiply(yTr, xTr)/ (1 + ewxy), axis=1).reshape(w.shape)
    #print(gradient.shape)
    # print(w.shape)
    #print(np.multiply(yTr,xTr).shape)
    #print(loss)
    return loss, gradient

    
    


    # ewxy = np.exp(-np.dot(np.dot(w.T, xTr), yTr.T))
    # # print(len(ewxy))
    # # print(len(ewxy[0]))
    # loss = np.log(1 + ewxy)
    # ewxy = np.exp(yTr*np.dot(w.T, xTr))
    # gradient = -np.sum((xTr * yTr) / (1 + ewxy))
    # return loss,gradient
