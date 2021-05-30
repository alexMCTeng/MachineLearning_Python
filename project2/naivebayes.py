#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayes(x, y, x1):
# =============================================================================
#function logratio = naivebayes(x,y,x1);
#
#Computation of log P(Y|X=x1) using Bayes Rule
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#x1: input vector of d dimensions (dx1)
#
#Output:
#logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
# =============================================================================


    
    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    X1= np.matrix(x1)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
# =============================================================================
# fill in code here

    # first calculate P(Y)
    py_pos, py_neg = naivebayesPY(X,y)
    # calculate P(X|Y)
    pxy_pos, pxy_neg = naivebayesPXY(x,y)
    # given x = X1, calculate P(x=X1|Y), given the count of the feature is 
    # x1, it should be p(w=a_i|Y)^x_i
    pxy_pos = np.prod(np.power(pxy_pos, X1))
    pxy_neg = np.prod(np.power(pxy_neg, X1))


    # calculate P(Y|X) = P(X|Y)P(Y) / P(X) for each class
    pyx_pos = pxy_pos * py_pos / (pxy_pos * py_pos + pxy_neg * py_neg)
    pyx_neg = pxy_neg * py_neg / (pxy_pos * py_pos + pxy_neg * py_neg)

    logratio = np.log(pyx_pos/pyx_neg)

    return logratio
# =============================================================================
