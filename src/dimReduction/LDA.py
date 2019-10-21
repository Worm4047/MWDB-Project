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
        # TOPICS_SIZE = 50
        # # h, w = self.dataMatrix.shape
        # print(self.dataMatrix)
        # imageFvs = self.dataMatrix.reshape((self.dataMatrix.shape[0] * self.dataMatrix.shape[1], self.dataMatrix.shape[2]))
        # kmeans = MiniBatchKMeans(n_clusters=TOPICS_SIZE, init='k-means++', batch_size=250, random_state=0,
        #                          verbose=0)
        # kmeans.fit(imageFvs)
        # kmeans.cluster_centers_
        # labels = kmeans.labels_
        # ldaDataMatrix = np.zeros((self.dataMatrix.shape[0], TOPICS_SIZE))
        # for imageIndex in range(self.dataMatrix.shape[0]):
        #     imageLabels = labels[imageIndex*self.dataMatrix.shape[1]: imageIndex*self.dataMatrix.shape[1] + self.dataMatrix.shape[1]]
        #     for label in imageLabels:
        #         ldaDataMatrix[imageIndex][label] += 1;
        #
        # print("Boom boom")

        # histogram of labels for each image = term-document matrix
        num_train_images = 18



        # A = np.zeros((TOPICS_SIZE, num_train_images))
        # ii = 0
        # jj = 0
        # for img_idx in range(num_train_images):
        #     if img_idx == 0:
        #         A[:, img_idx], bins = np.histogram(labels[0:num_kps],
        #                                            bins=range(TOPICS_SIZE + 1))
        #     else:
        #         ii = np.int(ii + num_kps)
        #         jj = np.int(ii + num_kps)
        #         A[:, img_idx], bins = np.histogram(labels[ii:jj], bins=range(TOPICS_SIZE + 1))
        #         # print str(ii) + ':' + str(jj)
        # end for
        # plt.figure()
        # plt.spy(A.T, cmap='gray')
        # plt.gca().set_aspect('auto')
        # plt.title('AP tf-idf corpus')
        # plt.xlabel('dictionary')
        # plt.ylabel('documents')
        # plt.show()

        # print(self.dataMatrix)
        lda_vb = LatentDirichletAllocation(n_components=self.k, max_iter=100, learning_method='online',
                                           batch_size=512,
                                           random_state=0, n_jobs=1)
        lda_vb.fit(self.dataMatrix)
        lda_vb.get_params()
        topics = lda_vb.components_
        H = lda_vb.transform(self.dataMatrix)

        return H, topics
