# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%
    w = w0
    ## << Insert your solution here
    loss, gradient = func(w0)
    ite = 0
    while ite <= maxiter and np.linalg.norm(gradient) >= tolerance:
        
        w = w - stepsize * gradient
        loss_new, gradient = func(w)
        ite = ite + 1

        if loss_new <= loss:
            stepsize = stepsize * 1.01
        else:
            stepsize = stepsize * 0.5
    # for i in range(1, maxiter):
    #     tmp = stepsize*func(w)[1]
    #     if tolerance > np.linalg.norm(w - tmp):
    #         break
    #     while func(w)[0] < func(w-tmp)[0]:
    #         stepsize = stepsize * 0.5
    #         tmp = stepsize*func(w)[1]
    #     w = w - stepsize * func(w)[1]
    #     stepsize = stepsize * 1.01

    ## >>    
    return w