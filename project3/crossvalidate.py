"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm
import random
from sklearn.model_selection import KFold


def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    train_errors = np.zeros((len(paras),len(Cs)))
    test_errors = np.zeros((len(paras),len(Cs)))

    kf = KFold(5, shuffle = False)

    for i in range(len(paras)):
        for j in range(len(Cs)):

            for train_index, test_index in kf.split(xTr.T): 

                xTr_train = xTr.T[train_index]
                yTr_train = yTr[train_index]
                xTr_test = xTr.T[test_index]
                yTr_test = yTr[test_index]

                svmclassify = trainsvm(xTr_train.T, yTr_train, Cs[j], ktype, paras[i])
            
                test_preds = svmclassify(xTr_test.T)
                test_errors[i, j] += np.mean(test_preds != yTr_test)/5

    print(test_errors)
    lowest_error = np.min(test_errors)
    x, y = np.where(test_errors == lowest_error)

    bestP = np.array(paras)[x[0]]
    bestC = np.array(Cs)[y[0]]
    errors = test_errors
    
    return bestC, bestP, lowest_error, errors


    