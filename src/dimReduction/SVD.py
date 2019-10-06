from src.dimReduction.interfaces.ReductionModel import ReductionModel
import numpy as np


class SVD(ReductionModel):
    def __init__(self, dataMatrix):
        super(SVD, self).__init__(dataMatrix)

    def getDecomposition(self):
        u, s, vt = np.linalg.svd(self.dataMatrix, full_matrices=False)
        return u, s, vt
