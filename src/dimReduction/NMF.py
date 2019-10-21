from src.dimReduction.interfaces.ReductionModel import ReductionModel
from sklearn.decomposition import NMF as NMF_SKLEARN
from sklearn import preprocessing

class NMF(ReductionModel):
    def __init__(self, dataMatrix, k=None, initialisation = 'random', tol = 1e-10, max_iter = 200):
        super(NMF, self).__init__(dataMatrix)
        self.k = k
        self.init = initialisation
        self.tol = tol
        self.max_iter = max_iter

    def getDecomposition(self):
        # refer https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        model = NMF_SKLEARN(n_components=self.k, init=self.init, tol=self.tol, max_iter=self.max_iter)
        W = model.fit_transform(self.dataMatrix)
        H = model.components_
        return W, H
