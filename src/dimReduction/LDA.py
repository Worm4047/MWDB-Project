from src.dimReduction.interfaces.ReductionModel import ReductionModel
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation


class LDA(ReductionModel):
    def __init__(self, dataMatrix, k=None):
        super(LDA, self).__init__(dataMatrix)
        self.k = k

    def getDecomposition(self):

        # cluster images into a dictionary
        # number has to be finalised after testing
        dictionary_size = 25
        h, w = self.dataMatrix.shape
        # print(w)
        # print(h)

        kmeans = MiniBatchKMeans(n_clusters=dictionary_size, init='k-means++', batch_size=250, random_state=0,
                                 verbose=0)
        kmeans.fit(self.dataMatrix)
        kmeans.get_params()
        kmeans.cluster_centers_
        labels = kmeans.labels_

        # histogram of labels for each image = term-document matrix
        num_train_images = h
        self.dataMatrix
        #num_kps needs to be calculated dynamically
        num_kps = 192
        A = np.zeros((dictionary_size, num_train_images))
        ii = 0
        jj = 0
        for img_idx in range(num_train_images):
            if img_idx == 0:
                A[:, img_idx], bins = np.histogram(labels[0:num_kps],
                                                   bins=range(dictionary_size + 1))
            else:
                ii = np.int(ii + num_kps)
                jj = np.int(ii + num_kps)
                A[:, img_idx], bins = np.histogram(labels[ii:jj], bins=range(dictionary_size + 1))
                # print str(ii) + ':' + str(jj)
        # end for
        # plt.figure()
        # plt.spy(A.T, cmap='gray')
        # plt.gca().set_aspect('auto')
        # plt.title('AP tf-idf corpus')
        # plt.xlabel('dictionary')
        # plt.ylabel('documents')
        # plt.show()

        # print(self.dataMatrix)

        # Needs to be finalised
        num_topics = 25

        lda_vb = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',
                                           batch_size=512,
                                           random_state=0, n_jobs=1)

        lda_vb.fit(self.dataMatrix.T)
        lda_vb.get_params()
        topics = lda_vb.components_
        H = lda_vb.transform(self.dataMatrix.T)

        # print(topics)
        # print(H.T)

        return topics, H.T
