import glob
import os
from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.common import latentSemanticsSaver
from src.common.dataMatrixHelper import read_data_matrix, save_data_matrix
from src.common.imageFeatureHelper import getImageFeatures

from src.common import latentSemanticsSaver
from src.dimReduction.LDA import LDA
from src.common.latentSemanticsSaver import getLatentSemanticPath

from src.common.imageHelper import getYUVImage, getGrayScaleImage
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

from src.dimReduction.PCA import PCA
from sklearn.cluster import MiniBatchKMeans


def getDataMatrix(imagePaths, modelType, label, directoryPath=None):
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

    # dataMatrix = read_data_matrix(modelType, label, "./store/dataMatrix/")
    dataMatrix = None
    if dataMatrix is None:
        dataMatrix = []
        if modelType == ModelType.CM:
            dataMatrix = getDataMatrixForCMForLDA(imagePaths, dataMatrix)
        if modelType == ModelType.LBP:
            dataMatrix = getDataMatrixForLBPForLDA(imagePaths, dataMatrix)
        if modelType == ModelType.HOG:
            dataMatrix = getDataMatrixForHOGForLDA(imagePaths, dataMatrix)
        if modelType == ModelType.SIFT:
            dataMatrix = getDataMatrixForSIFTForLDA(imagePaths, dataMatrix)
        # save_data_matrix_as_pickle(modelType, label, "./store/dataMatrix/", dataMatrix)
    return np.array(dataMatrix, np.float)


def getQueryImageRepList(vTranspose, imagePaths, modelType):
    featuresList = []
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Transforming Query Image | Processed {} out of {}".format(index, imagesCount))
        featuresList.append(getQueryImageRep(vTranspose, imagePath, modelType))
    print("featuresList:", featuresList)
    return np.array(featuresList)


def getQueryImageRepforLDA(vTranspose, imagePath, modelType, dimRedType):
    if not isinstance(modelType, ModelType):
        raise ValueError("Not a valid model type")

    if not isinstance(vTranspose, np.ndarray):
        raise ValueError("vTranspose should be a numpy array")

    if modelType == ModelType.CM:
        dataMatrix = getDataMatrixForCMForLDA([imagePath], [])
    elif modelType == ModelType.SIFT:
        dataMatrix = getDataMatrixForSIFTForLDA([imagePath], [])
    elif modelType == ModelType.LBP:
        dataMatrix = getDataMatrixForLBPForLDA([imagePath], [])
    elif modelType == ModelType.HOG:
        dataMatrix = getDataMatrixForHOGForLDA([imagePath], [])

    imageFeatures = dataMatrix[0]

    # imageFeatures = getImageFeatures(imagePath, modelType, True)

    # imageFeatures = tranformToLDAFeatures(imageFeatures)
    # print(vTranspose.shape)
    # print(imageFeatures.shape)
    if vTranspose.shape[1] != imageFeatures.shape[0]:
        raise ValueError("vTranspose dimensions are not matching with query image features")

    return np.array(np.matmul(imageFeatures, vTranspose.transpose()))


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

    return np.array(np.matmul(imageFeatures,vTranspose.transpose()))


# You may not need to call below methods
def getDataMatrixForCM(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeatures())
    return dataMatrix


def getDataMatrixForLBPForLDA(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dim = LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3).getFeatureWithDim()
        dataMatrix.append(dim)

    return tranformToLDAMatrix(np.array(dataMatrix))


def getDataMatrixForCMForLDA(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dim = ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeaturesWithDim()
        dim = dim.reshape((dim.shape[0] * dim.shape[1], dim.shape[2]))
        dataMatrix.append(dim)

    return tranformToLDAMatrix(np.array(dataMatrix))


def tranformToLDAMatrix(dataMatrix):
    TOPICS_SIZE = 50
    # h, w = self.dataMatrix.shape
    imageFvs = dataMatrix.reshape((dataMatrix.shape[0] * dataMatrix.shape[1], dataMatrix.shape[2]))
    kmeans = MiniBatchKMeans(n_clusters=TOPICS_SIZE, init='k-means++', batch_size=250, random_state=0,
                             verbose=0)
    kmeans.fit(imageFvs)
    kmeans.cluster_centers_
    labels = kmeans.labels_
    ldaDataMatrix = np.zeros((dataMatrix.shape[0], TOPICS_SIZE))
    for imageIndex in range(dataMatrix.shape[0]):
        imageLabels = labels[imageIndex * dataMatrix.shape[1]: imageIndex * dataMatrix.shape[1] +
                                                               dataMatrix.shape[1]]
        for label in imageLabels:
            ldaDataMatrix[imageIndex][label] += 1

    return ldaDataMatrix


def tranformToLDAFeatures(dataMatrix):
    TOPICS_SIZE = 50
    # h, w = self.dataMatrix.shape
    imageFvs = dataMatrix.reshape((dataMatrix.shape[0] * dataMatrix.shape[1], dataMatrix.shape[2]))
    print(imageFvs)
    kmeans = MiniBatchKMeans(n_clusters=TOPICS_SIZE, init='k-means++', batch_size=250, random_state=0,
                             verbose=0)
    kmeans.fit(imageFvs)
    kmeans.cluster_centers_
    labels = kmeans.labels_
    ldaDataMatrix = np.zeros((1, TOPICS_SIZE))
    for imageIndex in range(1):
        imageLabels = labels[imageIndex * dataMatrix.shape[1]: dataMatrix.shape[0]]
        for label in imageLabels:
            ldaDataMatrix[imageIndex][label] += 1

    return ldaDataMatrix


# def getDataMatrixForCMForLDA(imagePaths):
#     imagesCount = len(imagePaths)
#     rows = 0
#     for index, imagePath in enumerate(imagePaths):
#         if not os.path.exists(imagePath): continue
#         print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
#         features = ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeaturesWithDim()
#         # stack descriptors for all training images
#         if rows == 0:
#             matrix = features
#             rows += 1
#         else:
#             matrix = np.vstack((matrix, features))
#     return matrix

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


def getDataMatrixForSIFTForLDA(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    imageFeaturesCounts = []
    imageFeaturesStacked = []
    isFirst = True
    for index, imagePath in enumerate(imagePaths):
        if not os.path.exists(imagePath): continue
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        imageDes = SIFT(getGrayScaleImage(imagePath)).getFeatures()
        imageFeaturesCounts.append(len(imageDes))
        if isFirst:
            imageFeaturesStacked = imageDes
            isFirst = False
        else:
            imageFeaturesStacked = np.vstack((imageFeaturesStacked, imageDes))

    TOPICS_SIZE = 50
    # h, w = self.dataMatrix.shape
    kmeans = MiniBatchKMeans(n_clusters=TOPICS_SIZE, init='k-means++', batch_size=250, random_state=0,
                             verbose=0)
    kmeans.fit(imageFeaturesStacked)
    kmeans.cluster_centers_
    labels = kmeans.labels_
    ldaDataMatrix = np.zeros((imagesCount, TOPICS_SIZE))

    labelStartPointer = 0;
    for imageIndex, imageFeatureCount in enumerate(imageFeaturesCounts):
        for labelIndex in labels[labelStartPointer: labelStartPointer + imageFeatureCount]:
            ldaDataMatrix[imageIndex][labelIndex] += 1

        labelStartPointer += imageFeatureCount

    return ldaDataMatrix


# def getDataMatrixForLBPForLDA(imagePaths):
#     imagesCount = len(imagePaths)
#     rows = 0
#     for index, imagePath in enumerate(imagePaths):
#         if not os.path.exists(imagePath): continue
#         print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
#         features = LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3).getFeatures()
#         # stack descriptors for all training images
#         if rows == 0:
#             matrix = features
#             rows += 1
#         else:
#             matrix = np.vstack((matrix, features))
#     return matrix

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
        features = HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeaturesWithDim()
        features = features.reshape(
            (features.shape[0] * features.shape[1] * features.shape[2] * features.shape[3], features.shape[4]))
        dataMatrix.append(features)

    return tranformToLDAMatrix(np.array(dataMatrix))


# def getDataMatrixForHOGForLDA(imagePaths):
#     imagesCount = len(imagePaths)
#     rows = 0
#     for index, imagePath in enumerate(imagePaths):
#         if not os.path.exists(imagePath): continue
#         print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
#         features = HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeaturesWithDim()
#         # stack descriptors for all training images
#         if rows == 0:
#             matrix = features
#             rows += 1
#         else:
#             matrix = np.vstack((matrix, features))
#     return matrix

# def getDataMatrixForSIFTForLDA(imagePaths):
#     imagesCount = len(imagePaths)
#     rows = 0
#     for index, imagePath in enumerate(imagePaths):
#         if not os.path.exists(imagePath): continue
#         print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
#         features = SIFT(getGrayScaleImage(imagePath)).getFeatures()
#         # stack descriptors for all training images
#         if rows == 0:
#             matrix = features
#             rows += 1
#         else:
#             matrix = np.vstack((matrix, features))
#     return matrix

