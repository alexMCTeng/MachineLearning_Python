import numpy as np
from genTrainFeatures import genTrainFeatures
from naivebayesCL import naivebayesCL
from classifyLinear import classifyLinear


[x,y]=genTrainFeatures()
[w,b]=naivebayesCL(x,y)
preds=classifyLinear(x,w,b)
trainingerror=np.sum(preds!=y)/(y.shape[1])
print(trainingerror)