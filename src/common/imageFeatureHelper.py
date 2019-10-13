from src.models.enums.models import ModelType
from src.common.imageHelper import getGrayScaleImage, getYUVImage
from src.models.SIFT import SIFT
from src.models.ColorMoments import ColorMoments
from src.models.HOG import HOG
from src.models.LBP import LBP
import cv2
import numpy as np

def getImageFeatures(imagePath, modelType):
    if not isinstance(modelType, ModelType):
        raise ValueError("Invalid modelType")

    if modelType == ModelType.SIFT:
        return SIFT(getGrayScaleImage(imagePath)).getFeatures()
    if modelType == ModelType.CM:
        return ColorMoments(getYUVImage(imagePath), 100, 100).getFeatures()
    if modelType == ModelType.HOG:
        return HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeatures()
    if modelType == ModelType.LBP:
        return LBP(getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3).getFeatures()
