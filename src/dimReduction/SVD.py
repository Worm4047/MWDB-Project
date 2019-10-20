from src.dimReduction.interfaces.ReductionModel import ReductionModel
import numpy as np
from sklearn import preprocessing

class SVD(ReductionModel):
    def __init__(self, dataMatrix, k=None):
        super(SVD, self).__init__(dataMatrix)
        self.k = k

    def getDecomposition(self):
        self.dataMatrix = preprocessing.scale(self.dataMatrix)
        u, s, vt = np.linalg.svd(self.dataMatrix, full_matrices=False)
        s = np.diag(s)
        if self.k is None:
            return u, s, vt
        len = s.shape[0]
        # rank_s = np.linalg.matrix_rank(s)
        # s[(rank_s - self.k):rank_s] = 0
        for i in range(self.k, len):
           s[i, i] = 0
        u=np.matmul(u , s)
        uT = np.transpose(np.matmul(u, s))
        uT = uT[~np.all(uT == 0, axis=1)]
        u = np.transpose(uT)

        for i in range(self.k):
           s[i, i] = 1
        vt = np.matmul(s, vt)
        vt = vt[~np.all(vt == 0, axis=1)]

        return u, vt
