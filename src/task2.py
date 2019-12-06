import glob
import os
import csv
import numpy as np
import pandas as pd
import json
import scipy
import matplotlib
import networkx as nx
from src.dimReduction.dimRedHelper import DimRedHelper
from src.models.enums.models import ModelType
from src.common import comparisonHelper
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from scipy import linalg as LA
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import src.kmeans as kmeans

# This is a temp function and is used for testing only
class Task2():

    def __init__(self, dorsalImages, palmarImages, queryImages, dmDorsal, dmPalmar, dmqueryImages):
        self.palmarImages = palmarImages
        self.queryImages = queryImages
        self.dorsalImages = dorsalImages
        self.dmDorsal = dmDorsal
        self.dmPalmar = dmPalmar
        self.dmqueryImages = dmqueryImages
        
        #Computer Dorsal clusters
        self.dorsalClusters = self.getClusters(self.dmDorsal, self.dorsalImages, "dorsal")
        #Computer and save palmar clusters
        self.palmarClusters = self.getClusters(self.dmPalmar, self.palmarImages, "palmar")
        #Find labels for query images
        labels = self.getQueryLabels(self.dmqueryImages, self.dorsalClusters, self.palmarClusters)
        self.saveData(labels, self.queryImages, "query")

    def RbfKernel(self, data1, data2, sigma):
        delta =np.matrix(abs(np.subtract(data1, data2)))
        squaredEuclidean = (np.square(delta).sum(axis=1))
        result = np.exp(-(squaredEuclidean)/(2*sigma**2))
        return result

    def buildSimmilarityMatrix(self, d):
        nData = d.shape[0]
        result = np.matrix(np.full((nData,nData), 0, dtype=np.float))
        for i in range(0,nData):
            for j in range(0, nData):
                weight = self.RbfKernel(d[i, :], d[j, :], 0.4)
                result[i,j] = weight
        return result

    def buildDegreeMatrix(self, similarityMatrix):
        diag = np.array(similarityMatrix.sum(axis=1)).ravel()
        result = np.diag(diag)
        return result

    def unnormalizedLaplacian(self, simMatrix, degMatrix):
        result = degMatrix - simMatrix
        return result

    def transformToSpectral(self, laplacian, k):
        e_vals, e_vecs = LA.eig(np.matrix(laplacian))
        ind = e_vals.real.argsort()[:k]
        result = np.ndarray(shape=(laplacian.shape[0],0))
        for i in range(1, ind.shape[0]):
            cor_e_vec = np.transpose(np.matrix(e_vecs[:,np.asscalar(ind[i])]))
            result = np.concatenate((result, cor_e_vec), axis=1)
        return result

    def saveData(self, labels, images, name):
        imageLabelFileName = 'store/' + name + '_labels.json'
        label_dict = {}
        for i in range(len(labels)):
            label_dict[str(labels[i])] = []
        for i in range(len(labels)):
            label = str(labels[i])
            imagename = os.path.basename(images[i])
            imagepath = images[i]
            label_dict[label].append([imagename, imagepath])
        with open(imageLabelFileName, 'w+') as f:
            json.dump(label_dict, f)

        
    def getClusters(self, dmImages, images, name, c=5):

        # dmImages = dmImages.as_matrix()
        simMat = self.buildSimmilarityMatrix(dmImages)
        degMat = self.buildDegreeMatrix(simMat)
        lapMat = self.unnormalizedLaplacian(simMat, degMat)
        transformedData = self.transformToSpectral(lapMat, c)

        # kmeans = KMeans(n_clusters=c, random_state=0, init = 'k-means++').fit(transformedData)
        # out1 = kmeans.predict(transformedData)

        # labels = kmeans.labels_
        # centroids = kmeans.cluster_centers_
        print(transformedData.shape)
        centroids, labels = kmeans.kmeans(transformedData, c)

        print("Shape of clusters {}".format(len(centroids[0])))
        print("Shape of labels {}".format(len(labels)))
        label_dict = {}
        clust_count = {}
        clusters = []
        for i in range(len(dmImages)):
            cols = dmImages[i].shape[0]
            label = labels[i]
            if label not in clust_count:
                clust_count[label] = 0
            clust_count[label]+=1;
            if label not in label_dict:
                label_dict[label] = [0]*cols
            print(cols, len(label_dict[label]))
            for j in range(cols):
                label_dict[label][j] += dmImages[i][j]
        
        for key in label_dict:
            vals = label_dict[key]
            for i in range(len(vals)):
                vals[i] = vals[i]/clust_count[key]
            clusters.append(vals)
        
        self.saveData(labels, images, name)
        return clusters


    def getQueryLabels(self, dmQueries, dorsalClusters, palmarClusters):
        labels = []
        
        for dmQuery in dmQueries: 
            BestDorsalSim = self.getBestSim(dorsalClusters, dmQuery)
            BestPalmarSim = self.getBestSim(palmarClusters, dmQuery)
            print(BestDorsalSim, BestPalmarSim)
            if(BestDorsalSim > BestPalmarSim):
                labels.append("DORSAL")
            else:
                labels.append("PALMAR")
        return labels

    def getBestSim(self, centroids, dmQuery):
        sim = 0
        for centroid in centroids:
            sim += comparisonHelper.computeWithCosine(centroid,dmQuery)
        return sim/len(centroids)


def getLabelledDorsalImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, True)

def getLabelledPalmarImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, False)

def getLabelledImages(csvPath, imagePath, dorsal):
    label_df = pd.read_csv(csvPath)
    if dorsal:
        label_df = label_df.loc[ label_df['aspectOfHand'].str.contains('dorsal')]
    else:
        label_df = label_df.loc[ label_df['aspectOfHand'].str.contains('palmar')]
    images = list(label_df['imageName'].values)
    imagePaths = []
    for i in range(len(images)):
        imagePaths.append(os.path.join(imagePath, images[i]))
    return imagePaths

def getUnLabelledImages(csvPath, imagePath):
    label_df = pd.read_csv(csvPath)
    images = list(label_df['imageName'].values)
    imagePaths = []
    for i in range(len(images)):
        imagePaths.append(os.path.join(imagePath, images[i]))
    return imagePaths

def helper(csvpath, imagePath, queryPath, queryCsvPath):
# def helper():
#     csvpath = 'static/sample_data/labelled_set1.csv'
#     imagePath = 'static/sample_data/Labelled/Set1/'
#     queryPath = 'static/sample_data/Unlabelled/Set1/'
#     queryCsvPath = 'static/sample_data/unlabelled_set1.csv'
    
    # queryImage = [queryPath + 'Hand_0007735.jpg']
    dorsalImages = getLabelledDorsalImages(csvpath, imagePath)
    palmarImages = getLabelledPalmarImages(csvpath, imagePath)
    queryImages = getUnLabelledImages(queryCsvPath, queryPath)

    obj = DimRedHelper()

    dmDorsal = obj.getDataMatrixForLBP(dorsalImages, [])
    dmDorsal = np.stack( dmDorsal, axis=0 )

    dmPalmar = obj.getDataMatrixForLBP(palmarImages, [])
    dmPalmar = np.stack( dmPalmar, axis=0 )

    dmqueryImages = obj.getDataMatrixForLBP(queryImages, [])
    dmqueryImages = np.stack( dmqueryImages, axis=0 )

    obj = Task2(dorsalImages, palmarImages, queryImages, dmDorsal, dmPalmar, dmqueryImages)
    
if __name__ == '__main__':
    # helper()
    pass