from src.dimReduction.interfaces.ReductionModel import ReductionModel
from sklearn.decomposition import LatentDirichletAllocation


class LDA(ReductionModel):
    def __init__(self, dataMatrix, k=None):
        super(LDA, self).__init__(dataMatrix)
        self.k = k

    def getDecomposition(self):
        lda_vb = LatentDirichletAllocation(n_components=self.k, max_iter=100, learning_method='online',
                                           batch_size=512,
                                           random_state=0, n_jobs=1)
        lda_vb.fit(self.dataMatrix)
        lda_vb.get_params()
        topics = lda_vb.components_
        H = lda_vb.transform(self.dataMatrix)

        return H, topics
