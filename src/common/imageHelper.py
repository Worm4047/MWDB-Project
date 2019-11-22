from src.models.enums.models import ModelType
from src.models.SIFT import SIFT
from src.models.ColorMoments import ColorMoments
from src.models.HOG import HOG
from src.models.LBP import LBP
import cv2
import numpy as np
import cv2
import os
from src.constants import DATABASE_PATH

class ImageHelper:
    def getYUVImage(self, imagePath):
        return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YUV)

    def getGrayScaleImage(self, imagePath):
        return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    def getImageName(self, imagePath):
        return os.path.basename(imagePath).split('.')[0]

    def getRGBImage(self, imagePath):
        return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def getBGRImage(self, imagePath):
        return cv2.imread(imagePath, cv2.IMREAD_COLOR)

    def getRGBImageFromBGRImage(self, imagePath):
        return cv2.imread(imagePath, cv2.IMREAD_COLOR)

    def getImagePath(self, imageName):
        return os.path.join(DATABASE_PATH, imageName + ".jpg")

    def getImageFeatures(self, imagePath, modelType, retriveshape=False):
        if not isinstance(modelType, ModelType):
            raise ValueError("Invalid modelType")

        if (retriveshape):
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

    def draw_grid(self, img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA):
        '''(ndarray, 3-tuple, int, int) -> void
        draw gridlines on img
        line_color:
            BGR representation of colour
        thickness:
            line thickness
        type:
            8, 4 or cv2.LINE_AA
        pxstep:
            grid line frequency in pixels
        '''
        pxstep = 100
        pystep = 100
        x = pxstep
        y = pystep
        while x < img.shape[1]:
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
            x += pxstep

        while y < img.shape[0]:
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
            y += pystep

        return img