import numpy as np
from scipy.sparse.linalg import svds

def calcPCA(data,kv):
    cov_data = np.cov(data)
    u,s,v = svds(cov_data, k=kv, which ='LM')
    s = np.diag(s)
    u = np.flip(u,axis=1)
    s = np.flip(s)
    v = np.flip(v,axis=0) 
    return u,s,v