from src.dimReduction.NMF import NMF
from src.dimReduction.SVD import SVD
from src.models.enums.models import ModelType
import glob
import os
import cv2
from src.models.ColorMoments import ColorMoments
from src.models.SIFT import SIFT
from src.models.LBP import LBP
from src.models.HOG import HOG
from src.constants import BLOCK_SIZE
import numpy as np
from src.common.imageHelper import getYUVImage, getGrayScaleImage

from src.dimReduction.enums import reduction
from src.common.dataMatrixHelper import read_data_matrix, save_data_matrix
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from src.common.imageFeatureHelper import getImageFeatures
from src.common import latentSemanticsHelper

def getDataMatrix(imagePaths, modelType, directoryPath=None):
    imageFomat = "jpg"

    if modelType is None:
        raise ValueError("Arguments can not be null")

    if not isinstance(modelType, ModelType):
        raise ValueError("Invalid model type")

    if directoryPath is None and imagePaths is None:
        raise ValueError("Both directory path and image paths can not be None")

    if imagePaths is None:
        raise ValueError("Image paths need to be a list")

    if not isinstance(imagePaths, list) and not isinstance(imagePaths, np.ndarray):
        raise ValueError("Image paths need to be a iterable")

    if imagePaths is None:
        imagePaths = glob.glob(os.path.join(directoryPath, "*.{}".format(imageFomat)))

    dataMatrix = read_data_matrix(modelType, directoryPath)
    if dataMatrix is None:
        dataMatrix = []
        if modelType == ModelType.CM:
            getDataMatrixForCM(imagePaths, dataMatrix)
        if modelType == ModelType.LBP:
            getDataMatrixForLBP(imagePaths, dataMatrix)
        if modelType == ModelType.HOG:
            getDataMatrixForHOG(imagePaths, dataMatrix)
        if modelType == ModelType.SIFT:
            getDataMatrixForSIFT(imagePaths, dataMatrix)
        save_data_matrix(modelType, directoryPath, dataMatrix)
    return np.array(dataMatrix, dtype=np.float)

def getQueryImageRepList(vTranspose, imagePaths, modelType):
    featuresList = []
    for imagePath in imagePaths:
        featuresList.append(getQueryImageRep(vTranspose, imagePath, modelType))

    return np.array(featuresList)


def getQueryImageRep(vTranspose, imagePath, modelType):
    if not isinstance(modelType, ModelType):
        raise ValueError("Not a valid model type")

    if not isinstance(vTranspose, np.ndarray):
        raise ValueError("vTranspose should be a numpy array")

    imageFeatures = getImageFeatures(imagePath, modelType)
    if modelType == ModelType.SIFT:
        imageFeatures = getClusters(imageFeatures).flatten()

    if vTranspose.shape[1] != imageFeatures.shape[0]:
        raise ValueError("vTranspose dimensions are not matching with query image features")

    kSpaceRepresentation = []
    for row in vTranspose:
        kSpaceRepresentation.append(np.dot(row, imageFeatures))

    return np.array(kSpaceRepresentation)

# You may not need to call below methods
def getDataMatrixForCM(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeatures())
    return dataMatrix

def getClusters(descriptors):
    CLUSTERS_COUNT = 10
    wcss = []
    finalclusters = len(descriptors)
    kmeans = KMeans(n_clusters=CLUSTERS_COUNT, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(descriptors)
    # wcss.append(kmeans.inertia_)

    pointsCountMap = Counter(kmeans.labels_)
    pointsCountList = []
    for index in range(CLUSTERS_COUNT):
        # np.insert(kmeans.cluster_centers_[index], 0, pointsCountMap[index], axis=0)
        pointsCountList.append([pointsCountMap[index]/finalclusters])

    #Sort clusters in decreasing intra clusters distance

    return np.hstack((pointsCountList, kmeans.cluster_centers_))

def getDataMatrixForSIFT(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(getClusters(SIFT(getGrayScaleImage(imagePath)).getFeatures()).flatten())
    return dataMatrix

def getDataMatrixForLBP(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        lbp = LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3)
        features = lbp.getFeatures()
        dataMatrix.append(features)
    return dataMatrix

def getDataMatrixForHOG(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeatures())
    return dataMatrix

def getLatentSemantic(k, decompType, dataMatrix, modelType, label, imageDirName):
    folderName = "{}_{}_{}_{}_{}".format(imageDirName, modelType.name, decompType.name, k, label)
    latent_semantic = latentSemanticsHelper.getSemanticsFromFolder(folderName)
    if latent_semantic is None:
        if decompType == reduction.ReductionType.SVD:
            u, s, v = SVD(dataMatrix, k).getDecomposition()
            latent_semantic = u, v
        elif decompType == reduction.ReductionType.PCA:
            print "check later"
        #u, s, v = PCA(dataMatrix, k).getDecomposition()
        elif decompType == reduction.ReductionType.NMF:
            latent_semantic = NMF(dataMatrix, k).getDecomposition()
        else:
            print "check later"
        latentSemanticsHelper.saveSemantics(imageDirName, modelType, label, decompType, k, latent_semantic[0], latent_semantic[1])
    return latent_semantic