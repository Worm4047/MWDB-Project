import os
import pandas as pd
from src.constants import DATABASE_PATH, IMAGE_IDS_CSV, DORSAL_DATABASE_PATH, PALMAR_DATABASE_PATH, DORSAL_IMAGE_IDS_CSV, PALMAR_IMAGE_IDS_CSV
from src.common.imageHelper import ImageHelper
from src.classifiers.pprClassifier import ImageClass
import glob
import numpy as np

class ImageIndex:
    imageNameToIdDf = None
    imageIdToNameDf = None
    imageHelper = None
    databasePath = None
    imageIdsCSVPath = None
    imageClass = None

    def __init__(self, imageClass=None):
        self.imageClass = imageClass
        self.setDatabasePath()
        self.setImageIdsCSVPath()

        self.imageNameToIdDf = self.getImageIdsFromFile()
        self.imageIdToNameDf = self.imageNameToIdDf.copy()

        self.imageNameToIdDf.set_index("imageName", inplace=True)
        self.imageIdToNameDf.set_index("imageId", inplace=True)
        self.imageHelper = ImageHelper()

    def setImageIdsCSVPath(self):
        if self.imageClass == ImageClass.DORSAL:
            self.imageIdsCSVPath = DORSAL_IMAGE_IDS_CSV
        if self.imageClass == ImageClass.PALMAR:
            self.imageIdsCSVPath = PALMAR_IMAGE_IDS_CSV
        if self.imageClass is None:
            self.imageIdsCSVPath = IMAGE_IDS_CSV

    def setDatabasePath(self):
        if self.imageClass == ImageClass.DORSAL:
            self.databasePath = DORSAL_DATABASE_PATH
        if self.imageClass == ImageClass.PALMAR:
            self.databasePath = PALMAR_DATABASE_PATH
        if self.imageClass is None:
            self.databasePath = DATABASE_PATH

    def createImageIdsFile(self):
        imagePaths = glob.glob(os.path.join(self.databasePath, "*.jpg"))
        imageDfDict = {
            "imageName": [],
            "imageId": []
        }
        for index, imagePath in enumerate(imagePaths):
            imageDfDict["imageName"].append(os.path.basename(imagePath).split('.')[0])
            imageDfDict["imageId"].append(index)

        imageIdDf = pd.DataFrame.from_dict(imageDfDict)
        imageIdDf.to_csv(self.imageIdsCSVPath)
        return imageIdDf

    def getImageIdsFromFile(self):
        if not os.path.exists(self.imageIdsCSVPath):
            return self.createImageIdsFile()

        return pd.read_csv(self.imageIdsCSVPath)

    def getImageName(self, imageId):
        return self.imageIdToNameDf.loc[imageId, 'imageName']

    def getImageId(self, imageName):
        return int(self.imageNameToIdDf.loc[imageName, 'imageId'])

    def getImageIdForPath(self, imagePath):
        return int(self.imageNameToIdDf.loc[self.imageHelper.getImageName(imagePath), 'imageId'])

    def getImageIds(self, imagePaths):
        return [self.getImageId(self.imageHelper.getImageName(imagePath)) for imagePath in imagePaths]

    def getImagePaths(self, imageIds):
        return [self.imageHelper.getImagePath(self.getImageName(imageId)) for imageId in imageIds]

    def deleteIndexFile(self):
        os.remove(self.imageIdsCSVPath)


