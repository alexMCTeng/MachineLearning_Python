import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))
    
    D = euclidean_distances(X.T, Y=Z.T)


    # for i in range(n):
    #     for j in range(m):
    #         diff = X[:, i] - Z[:, j]
    #         D[i, j] = np.sqrt(np.dot(diff.T, diff))
    return D
