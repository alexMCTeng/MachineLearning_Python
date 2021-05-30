
import numpy as np
from numpy import linalg as la
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-03):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    loss = float('inf')
    w = w0
    for i in range(maxiter):
        loss_t = loss
        loss, gradient = func(w)
        if (la.norm(gradient) < tolerance):
            break
        if loss <= loss_t:
            stepsize = 1.01 * stepsize
        else:
            stepsize = 0.5 * stepsize
        if stepsize < eps:
            break
        w = w - stepsize * gradient
        
    return w
