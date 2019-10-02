from src.models.enums.models import ModelType
import glob
import os
import cv2
from src.models.ColorMoments import ColorMoments
from src.models.SIFT import SIFT
from src.models.LBP import LBP
from src.constants import BLOCK_SIZE
import numpy as np
from src.common.imageHelper import getYUVImage, getGrayScaleImage

def getDataMatrix(directoryPath, modelType):
    imageFomat = "jpg"

    if directoryPath is None or modelType is None:
        raise ValueError("Arguments can not be null")

    if not isinstance(modelType, ModelType):
        raise ValueError("Invalid model type")

    imagePaths = glob.glob(os.path.join(directoryPath, "*.{}".format(imageFomat)))
    dataMatrix = []
    if modelType == ModelType.CM:
        getDataMatrixForCM(imagePaths, dataMatrix)
    if modelType == ModelType.LBP:
        getDataMatrixForLBP(imagePaths, dataMatrix)
    if modelType == ModelType.HOG:
        getDataMatrixForCM(imagePaths, dataMatrix)
    # if modelType == ModelType.SIFT:
    #     getDataMatrixForSIFT(imagePaths, dataMatrix)

    return np.array(dataMatrix, dtype=np.float)

def getDataMatrixForCM(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(ColorMoments(getYUVImage(imagePath), BLOCK_SIZE, BLOCK_SIZE).getFeatures())
    return dataMatrix

def getDataMatrixForSIFT(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        dataMatrix.append(SIFT(getGrayScaleImage(imagePath)).getFeatures())
    return dataMatrix

def getDataMatrixForLBP(imagePaths, dataMatrix):
    imagesCount = len(imagePaths)
    for index, imagePath in enumerate(imagePaths):
        print("Data matrix creation | Processed {} out of {} images".format(index, imagesCount - 1))
        lbp = LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3)
        features = lbp.getFeatures()
        dataMatrix.append(features)
    return dataMatrix
