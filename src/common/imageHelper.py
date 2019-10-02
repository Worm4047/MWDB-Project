import cv2
import os

def getYUVImage(imagePath):
    return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YUV)

def getGrayScaleImage(imagePath):
    return cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

def getImageName(imagePath):
    return os.path.basename(imagePath)
