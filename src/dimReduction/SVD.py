from src.dimReduction.interfaces.ReductionModel import ReductionModel
import numpy as np

class SVD(ReductionModel):
    def __init__(self, dataMatrix, k=None):
        super(SVD, self).__init__(dataMatrix)
        self.k = k

    def getDecomposition(self):
        u, s, vt = np.linalg.svd(self.dataMatrix, full_matrices=False)
        s = np.diag(s)
        if self.k is None:
            return u, s, vt
        len = s.shape[0]
        for i in range(self.k, len):
            s[i, i] = 0

        uT = np.transpose(np.matmul(u, s))
        uT = uT[~np.all(uT == 0, axis=1)]
        u = np.transpose(uT)

        vt = np.matmul(s, vt)
        vt = vt[~np.all(vt == 0, axis=1)]

        return u, s, vt
