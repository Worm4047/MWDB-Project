import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA as PCA_SKLEARN
from src.dimReduction.interfaces.ReductionModel import ReductionModel
from sklearn import preprocessing

class PCA(ReductionModel):
    def __init__(self, dataMatrix, k=None):
        super(PCA, self).__init__(dataMatrix)
        self.k = k

    def getDecomposition(self):
        pca = PCA_SKLEARN(n_components=self.k)
        self.dataMatrix = preprocessing.scale(self.dataMatrix)
        u = pca.fit_transform(self.dataMatrix)
        v = pca.components_
        return u, v

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

