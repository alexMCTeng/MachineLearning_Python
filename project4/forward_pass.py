# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:30:29 2019

@author: remus
"""
import numpy as np
def forward_pass(W, xTr, trans_func):
#% function [as,zs]=forward_pass(W,xTr,trans_func)
#%
#% INPUT:
#% W weights (list of numpy array)
#% xTr dxn numpy array (each column is an input vector)
#% trans_func transition function to apply for inner layers
#%
#% OUTPUTS:
#%
#% as = result of forward pass 
#% zs = result of forward pass (zs[0] output layer of the forward pass) 
#%
    n = np.shape(xTr)[1]
    
    ## CHECK!  -JERRY
    
    # First, we add the constant weight
    zzs = [None]*(len(W)+1);   zzs[-1] = np.vstack((xTr, np.ones([1, n])))
    aas = [None]*(len(W)+1);   aas[-1] = xTr
    # print('w shape', np.array(W).shape)
    # print('zzs shape', zzs[-1].shape) [14, 305], [14, 101]
    # print('aas shape', len(aas))
    # print('zzs', zzs[0])  None
    
    # Do the forward process here
    for i in range(len(W)-1, 0, -1):   # len(W) = 4, 4 hidden layers
        # INSERT CODE
        #<<kqw
        # aas[i] = W[i] @ zzs[i+1]
        aas[i] = np.dot(W[i], zzs[i+1])
        # print(W[i].shape)     # (1, 21) (20, 14) (20, 21) (20, 21), number of hidden nodes * d
        # print('aas[i]', aas[i].shape)   #(20, 305) (20, 305) (20, 305) (1, 305) (20, 101) (20, 101) (20, 101) (1, 101)
        # zzs[i+1],   (14, 305) (21, 305) (21, 305) (21, 305) (14, 101) (21, 101) (21, 101) (21, 101)
        zzs[i] = np.vstack((trans_func(aas[i]), np.ones((1,n))))
        
    # INSERT CODE: (last one is special, no transition function)
    ##<<kqw
    # zzs[0]=W[0]@zzs[1]
    zzs[0] = np.dot(W[0], zzs[1])
    aas[0] = zzs[0]
    ##>>kqwend
    
    return aas, zzs
