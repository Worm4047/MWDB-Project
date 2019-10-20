import glob
import os
from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.common import latentSemanticsHelper
from src.common.dataMatrixHelper import read_data_matrix, save_data_matrix
from src.common.imageFeatureHelper import getImageFeatures

from src.common import latentSemanticsHelper
from src.dimReduction.LDA import LDA
from src.common.latentSemanticsHelper import getLatentSemanticPath

from src.common.imageHelper import getYUVImage, getGrayScaleImage
from src.constants import BLOCK_SIZE
from src.dimReduction.LDA import LDA
from src.dimReduction.NMF import NMF
from src.dimReduction.SVD import SVD
from src.dimReduction.enums import reduction
from src.models.ColorMoments import ColorMoments
from src.models.HOG import HOG
from src.models.LBP import LBP
from src.models.SIFT import SIFT
from src.models.enums.models import ModelType

from src.dimReduction.PCA import PCA


def getDataMatrix(imagePaths, modelType, label, directoryPath=None):
    imageFomat = "jpg"
    print(directoryPath)
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

    dataMatrix = read_data_matrix(modelType, label, "./store/dataMatrix/")
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
        save_data_matrix(modelType, label, "./store/dataMatrix/", dataMatrix)
    return np.array(dataMatrix, dtype=np.float)

def getDataMatrixForLDA(imagePaths, modelType, label, directoryPath=None):
    imageFomat = "jpg"
    print(directoryPath)
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

    dataMatrix = read_data_matrix(modelType, label, "./store/dataMatrix/")
    if dataMatrix is None:
        dataMatrix = []
        if modelType == ModelType.CM:
            getDataMatrixForCMForLDA(imagePaths, dataMatrix)
        if modelType == ModelType.LBP:
            getDataMatrixForLBPForLDA(imagePaths, dataMatrix)
        if modelType == ModelType.HOG:
            getDataMatrixForHOGForLDA(imagePaths, dataMatrix)
        if modelType == ModelType.SIFT:
            raise ValueError("SIFT can not be called with LDA")
        save_data_matrix(modelType, label, "./store/dataMatrix/", dataMatrix)
    return np.array(dataMatrix, dtype=np.float)

def getQueryImageRepList(vTranspose, imagePaths, modelType):
    featuresList = []
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Transforming Query Image | Processed {} out of {}".format(index, imagesCount))
        featuresList.append(getQueryImageRep(vTranspose, imagePath, modelType))
    print("featuresList:",featuresList)
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
    #for row in vTranspose:
        #kSpaceRepresentation.append(np.dot(row, imageFeatures))
    #    kSpaceRepresentation.append(np.matmul(imageFeatures,row))
    return np.array(np.matmul(imageFeatures,vTranspose.transpose()))


# You may not need to call below methods
def getDataMatrixForCM(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeatures())
    return dataMatrix

def getDataMatrixForCMForLDA(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeaturesWithDim())
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
        pointsCountList.append([pointsCountMap[index] / finalclusters])

    # Sort clusters in decreasing intra clusters distance

    return np.hstack((pointsCountList, kmeans.cluster_centers_))


def getDataMatrixForSIFT(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(getClusters(SIFT(getGrayScaleImage(imagePath)).getFeatures()).flatten())
    return dataMatrix

def getDataMatrixForLBP(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        lbp = LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3)
        features = lbp.getFeatures()
        dataMatrix.append(features)
    return dataMatrix

def getDataMatrixForLBPForLDA(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        lbp = LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3)
        features = lbp.getFeatureWithDim()
        dataMatrix.append(features)
    return dataMatrix

def getDataMatrixForHOG(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeatures())
    return dataMatrix

def getDataMatrixForHOGForLDA(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeaturesWithDim())
    return dataMatrix


def getLatentSemantic(k, decompType, dataMatrix, modelType, label, imageDirName, imagePaths):
    folderName = "{}_{}_{}_{}_{}".format(imageDirName, modelType.name, decompType.name, k, label)
    lsPath = getLatentSemanticPath(os.path.basename(imageDirName), modelType, decompType, k, label)
    latent_semantic = latentSemanticsHelper.getSemanticsFromFolder(lsPath)
    if latent_semantic is None:
        if decompType == reduction.ReductionType.SVD:
            u, v = SVD(dataMatrix, k).getDecomposition()
            latent_semantic = u, v
        elif decompType == reduction.ReductionType.PCA:
            latent_semantic = PCA(dataMatrix, k).getDecomposition()
        elif decompType == reduction.ReductionType.NMF:
            latent_semantic = NMF(dataMatrix, k).getDecomposition()
        elif decompType == reduction.ReductionType.LDA:
            latent_semantic = LDA(dataMatrix, k).getDecomposition()
        else:
            print("Check later")
            return None
        print("Image path example ", imagePaths[0])
        latentSemanticsHelper.saveSemantics(os.path.basename(imageDirName), modelType, label, decompType, k, latent_semantic[0], latent_semantic[1], imagePaths=imagePaths)
    return latent_semantic
