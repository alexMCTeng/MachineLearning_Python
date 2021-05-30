#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#    
#    Output:
#    posprob: probability vector of p(x|y=1) (dx1)
#    negprob: probability vector of p(x|y=-1) (dx1)
# =============================================================================


    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    Y = np.matrix(y)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.matrix('-1, 1')
    
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
        #Xnew = np.concatenate((X, X0), axis=1) #concatenate to column
    Ynew = np.hstack((Y, Y0))
    # print("Y matrix: ",Ynew)
    # print("X matrix: ", Xnew)
    
    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    
# =============================================================================
# fill in code here

    #ytemp = (Ynew==1)

    # calculate sum_{i=1}^{d} [x_a]_i
    
    # sum_xi = np.sum(Xnew, axis=0)

    # from my opinion, we do not have to add a smoothing paramter
    # as we already add one all-ones positive and negative example.
    # use multinominal case
    # posprob = np.sum(np.multiply((Ynew == 1), Xnew), axis=1) / np.sum(np.multiply((Ynew == 1), sum_xi), axis=1)
    # negprob = np.sum(np.multiply((Ynew == -1), Xnew), axis=1) / np.sum(np.multiply((Ynew == -1), sum_xi), axis=1)
    
    # pos = np.where(Ynew == 1)
    # posprob = np.divide(np.sum(Xnew[:, pos[1]], axis=1), np.sum(np.sum(Xnew[:, pos[1]])))

    # pos = np.where(Ynew == -1)
    # negprob = np.divide(np.sum(Xnew[:, pos[1]], axis=1), np.sum(np.sum(Xnew[:, pos[1]])))



    posprob = np.sum(np.multiply((Ynew == 1), (Xnew==1)), axis=1) / np.sum((Ynew == 1), axis=1)
    negprob = np.sum(np.multiply((Ynew == -1), (Xnew==1)), axis=1) / np.sum((Ynew == -1), axis=1)


    return posprob,negprob

# =============================================================================
