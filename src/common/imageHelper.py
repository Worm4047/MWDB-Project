from src.models.enums.models import ModelType
from src.models.SIFT import SIFT
from src.models.ColorMoments import ColorMoments
from src.models.HOG import HOG
from src.models.LBP import LBP
import cv2
import numpy as np
import cv2
import os

class ImageHelper:
    def getYUVImage(self, imagePath):
        return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YUV)

    def getGrayScaleImage(self, imagePath):
        return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    def getImageName(self, imagePath):
        return os.path.basename(imagePath)

    def getImageFeatures(self, imagePath, modelType, isLDA=False):
        if not isinstance(modelType, ModelType):
            raise ValueError("Invalid modelType")

        if (isLDA):
            if modelType == ModelType.SIFT:
                return SIFT(self.getGrayScaleImage(imagePath)).getFeatures()
            if modelType == ModelType.CM:
                return ColorMoments(self.getYUVImage(imagePath), 100, 100).getFeaturesWithDim()
            if modelType == ModelType.HOG:
                return HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeaturesWithDim()
            if modelType == ModelType.LBP:
                return LBP(self.getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3).getFeatureWithDim()
        else:
            if modelType == ModelType.SIFT:
                return SIFT(self.getGrayScaleImage(imagePath)).getFeatures()
            if modelType == ModelType.CM:
                return ColorMoments(self.getYUVImage(imagePath), 100, 100).getFeatures()
            if modelType == ModelType.HOG:
                return HOG(cv2.imread(imagePath, cv2.IMREAD_COLOR), 9, 8, 2).getFeatures()
            if modelType == ModelType.LBP:
                return LBP(self.getGrayScaleImage(imagePath), blockSize=100, numPoints=24, radius=3).getFeatures()

