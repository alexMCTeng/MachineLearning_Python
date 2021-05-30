"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    
    valid = (C - alphas) * alphas
    index = np.where(valid == np.max(valid))
    K_new = K[:, index[0]]
    y_new = yTr[index[0], :]

    bias = 1 / y_new - np.sum(yTr * alphas * K_new)
  
    
    return bias 
    
