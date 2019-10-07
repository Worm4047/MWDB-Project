
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.neighbors import KDTree
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA

from src.dimReduction.interfaces.ReductionModel import ReductionModel

class LDA(ReductionModel):
    def __init__(self, dataMatrix):
        super(LDA, self).__init__(dataMatrix)
        self.dictionary_size = 20
        self.numtopics = 8
        self.batch_size = 50
        # assuming 
        self.train_images = dataMatrix.shape[1]
        self.lenkp = 192

    def getDecomposition(self):
        #cluster images into a dictionary
        
        kmeans = MiniBatchKMeans(n_clusters = self.dictionary_size, batch_size = self.batch_size, random_state=0)
        kmeans.fit(self.dataMatrix)
        # kmeans.get_params()
        # kmeans.cluster_centers_
        labels = kmeans.labels_
        #histogram of labels for each image = term-document matrix
        A = np.zeros((self.dictionary_size,self.train_images))
        
        ii = 0
        jj = 0
        for img_idx in range(self.train_images):
            if img_idx == 0:
                A[:,img_idx], bins = np.histogram(labels[0:int(self.lenkp)], bins=range(self.dictionary_size+1))
            # print(img_idx, num_kps[img_idx])
            else:
                ii = np.int(ii + self.lenkp)
                jj = np.int(ii + self.lenkp)
                A[:,img_idx], bins = np.histogram(labels[ii:jj] , bins=range(self.dictionary_size+1))
            #end for

        plt.figure()
        plt.spy(A.T, cmap = 'gray')   
        plt.gca().set_aspect('auto')
        plt.title('AP tf-idf corpus')
        plt.xlabel('dictionary')
        plt.ylabel('documents')    
        plt.show() 
        
        #fit LDA topic model based on tf-idf of term-document matrix
        num_features = self.dictionary_size
        # num_topics = self.numtopics #fixed for LDA
                    
        #fit LDA model
        print("Fitting LDA model...", self.numtopics)
        lda_vb = LatentDirichletAllocation(n_components = self.numtopics, max_iter=10, learning_method='online', batch_size = 512, random_state=0, n_jobs=1)

        print(type(A))
        
        lda_vb.fit(A.T)  #online VB

        print("LDA params")
        print(lda_vb.get_params())

        #topic matrix W: K x V
        #components[i,j]: topic i, word j
        #note: here topics correspond to label clusters
        
        #---------------------
        topics = lda_vb.components_
        #----------------------
        f = plt.figure()
        plt.matshow(topics, cmap = 'gray')   
        plt.gca().set_aspect('auto')
        plt.title('learned topic matrix')
        plt.ylabel('topics')
        plt.xlabel('dictionary')
        plt.show()
        
        #topic proportions matrix: D x K
        #note: np.sum(H, axis=1) is not 1
        ############################
        H = lda_vb.transform(A.T)
        #############################
        f = plt.figure()
        plt.matshow(H, cmap = 'gray')   
        plt.gca().set_aspect('auto')
        plt.show()
        plt.title('topic proportions')
        plt.xlabel('topics')
        plt.ylabel('documents')
        plt.show()