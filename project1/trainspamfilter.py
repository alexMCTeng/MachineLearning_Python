
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io

def trainspamfilter(xTr,yTr):

    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!

    f = lambda w : ridge(w,xTr,yTr,0.001)
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-04,1000)
    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained

    # ridge lambda = 0.02, step = 1e-05, 20000 iter, tolerance = 1e-04 98.76%
    # hinge lambda = 0.002, step = 1e-05, 30000 iter, tolerance = 1e-04
    # logsitic , step = 1e-04, 2000 iter= 20000, tolerance = 1e-04, 98.38%
