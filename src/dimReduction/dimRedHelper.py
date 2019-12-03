import glob
import os
from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.common import latentSemanticsSaver
from src.common.dataMatrixHelper import read_data_matrix, save_data_matrix
from src.constants import BLOCK_SIZE
from src.dimReduction.LDA import LDA
from src.dimReduction.NMF import NMF
from src.dimReduction.SVD import SVD
from src.dimReduction.enums import reduction
from src.dimReduction.enums.reduction import ReductionType
from src.models.ColorMoments import ColorMoments
from src.models.HOG import HOG
from src.models.LBP import LBP
from src.models.SIFT import SIFT
from src.models.enums.models import ModelType
from src.common.imageHelper import ImageHelper
from src.models.featureArchiver import FeatureArchiver


class DimRedHelper:
    featureArchiver = None

    def __init__(self):
        self.featureArchiver = FeatureArchiver()
        pass

    def getDataMatrix(self, imagePaths, modelType, label=None, directoryPath=None):
        imageFomat = "jpg"
        # print(directoryPath)
        if modelType is None:
            raise ValueError("Arguments can not be null")
        elif not isinstance(modelType, ModelType):
            raise ValueError("Invalid model type")
        elif imagePaths is None and directoryPath is None:
            raise ValueError("Both directory path and image paths can not be None")
        elif directoryPath is None and not isinstance(imagePaths, list) and not isinstance(imagePaths, np.ndarray):
            raise ValueError("Image paths need to be a iterable")

        if imagePaths is None:
            imagePaths = glob.glob(os.path.join(directoryPath, "*.{}".format(imageFomat)))

        # dataMatrix = read_data_matrix(modelType, label, "./store/dataMatrix/")
        dataMatrix = None
        if dataMatrix is None:
            dataMatrix = []
            if modelType == ModelType.CM:
                self.getDataMatrixForCM(imagePaths, dataMatrix)
            if modelType == ModelType.LBP:
                self.getDataMatrixForLBP(imagePaths, dataMatrix)
            if modelType == ModelType.HOG:
                self.getDataMatrixForHOG(imagePaths, dataMatrix)
            if modelType == ModelType.SIFT:
                self.getDataMatrixForSIFT(imagePaths, dataMatrix)
            # save_data_matrix(modelType, label, "./store/dataMatrix/", dataMatrix)
        return np.array(dataMatrix, dtype=np.float)

    def getDataMatrixForCM(self, imagePaths, dataMatrix):
        imagesCount = len(imagePaths)
        for index, imagePath in enumerate(imagePaths):
            try:
                dataMatrix.append(self.featureArchiver.getFeaturesForImage(imagePath, modelType=ModelType.CM).flatten())
            except:
                if not os.path.exists(imagePath): continue
                print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
                dataMatrix.append(ColorMoments(ImageHelper().getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeatures())
        return dataMatrix

    def getDataMatrixForLBP(self, imagePaths, dataMatrix):
        imagesCount = len(imagePaths)
        for index, imagePath in enumerate(imagePaths):
            try:
                dataMatrix.append(self.featureArchiver.getFeaturesForImage(imagePath, modelType=ModelType.CM).flatten())
            except:
                if not os.path.exists(imagePath): continue
                print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
                lbp = LBP(ImageHelper().getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3)
                features = lbp.getFeatures()
                dataMatrix.append(features)
        return dataMatrix

    def getDataMatrixForHOG(self, imagePaths, dataMatrix):
        imagesCount = len(imagePaths)
        for index, imagePath in enumerate(imagePaths):
            try:
                dataMatrix.append(self.featureArchiver.getFeaturesForImage(imagePath, modelType=ModelType.HOG).flatten())
            except:
                if not os.path.exists(imagePath): continue
                print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
                dataMatrix.append(HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeatures())
        return dataMatrix

    def getClusters(self, descriptors):
        CLUSTERS_COUNT = 20
        wcss = []
        finalclusters = len(descriptors)
        kmeans = KMeans(n_clusters=CLUSTERS_COUNT, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(descriptors)
        # wcss.append(kmeans.inertia_)

        pointsCountMap = Counter(kmeans.labels_)
        pointsCountList = []
        for index in range(CLUSTERS_COUNT):
            # np.insert(kmeans.cluster_centers_[index], 0, pointsCountMap[index], axis=0)
            pointsCountList.append([pointsCountMap[index] / finalclusters])

        # Sort clusters in decreasing intra clusters distance

        return np.hstack((pointsCountList, kmeans.cluster_centers_))

    def getDataMatrixForSIFT(self, imagePaths, dataMatrix):
        imagesCount = len(imagePaths)
        for index, imagePath in enumerate(imagePaths):
            if not os.path.exists(imagePath): continue
            print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
            dataMatrix.append(self.getClusters(SIFT(ImageHelper().getGrayScaleImage(imagePath)).getFeatures()).flatten())
        return dataMatrix

