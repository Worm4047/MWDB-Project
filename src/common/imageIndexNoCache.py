import os
import pandas as pd
from src.constants import DATABASE_PATH, IMAGE_IDS_CSV, DORSAL_DATABASE_PATH, PALMAR_DATABASE_PATH, DORSAL_IMAGE_IDS_CSV, PALMAR_IMAGE_IDS_CSV
from src.common.imageHelper import ImageHelper
from src.classifiers.pprClassifier import ImageClass
import glob
import numpy as np

class ImageIndexNoCache:
    imageHelper = None
    imagePaths = None
    imageClasses = None
    imagePathToImageIdMap = {}

    def __init__(self, imagePaths, imageClasses):
        self.imageHelper = ImageHelper()
        self.imagePaths = imagePaths
        self.imageClasses = imageClasses
        self.imagePathToImageIdMap = {}
        for index, imagePath in enumerate(imagePaths):
            self.imagePathToImageIdMap[imagePath] = index

    def getImageName(self, imageId):
        return self.imageHelper.getImageName(self.imagePaths[imageId])

    def getImagePathForImageId(self, imageId):
        return self.imagePaths[imageId]

    def getImageIdForImagePath(self, imagePath):
        return self.imagePathToImageIdMap[imagePath]

    def getImageIdForPath(self, imagePath):
        return self.imagePathToImageIdMap[imagePath]

    def getImageIdsForPaths(self, imagePaths):
        return [self.getImageIdForImagePath(imagePath) for imagePath in imagePaths]

    def getImageClassForImageId(self, imageId):
        return self.imageClasses[imageId]

    def getImageClassForImageIds(self, imageIds):
         return [self.getImageClassForImageId(imageId) for imageId in imageIds]

    def getImagePathsForImageIds(self, imageIds):
        return [self.getImagePathForImageId(imageId) for imageId in imageIds]
