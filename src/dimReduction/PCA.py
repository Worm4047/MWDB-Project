import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA



def calcPCA(data,kv):
    """print("data = ", data.shape)
    cov_data = np.cov(data)
    print("covv shape = ", cov_data.shape)
    u,s,v = svds(cov_data, k=kv, which ='LM')
    print("v shape = ", v.shape)
    s = np.diag(s)
    u = np.flip(u,axis=1)
    s = np.flip(s)
    v = np.flip(v,axis=0) """
    pca = PCA(n_components=kv)
    u = pca.fit_transform(data)
    v = pca.components_
    return u,v

