import os
import pandas as pd
from src.constants import DATABASE_PATH, IMAGE_IDS_CSV
from src.common.imageHelper import ImageHelper
import glob
import numpy as np

class ImageIndex:
    imageNameToIdDf = None
    imageIdToNameDf = None
    imageHelper = None

    def __init__(self):
        self.imageNameToIdDf = self.getImageIdsFromFile()
        self.imageIdToNameDf = self.imageNameToIdDf.copy()

        self.imageNameToIdDf.set_index("imageName", inplace=True)
        self.imageIdToNameDf.set_index("imageId", inplace=True)
        self.imageHelper = ImageHelper()

    def createImageIdsFile(self):
        imagePaths = glob.glob(os.path.join(DATABASE_PATH, "*.jpg"))
        imageDfDict = {
            "imageName": [],
            "imageId": []
        }
        for index, imagePath in enumerate(imagePaths):
            imageDfDict["imageName"].append(os.path.basename(imagePath).split('.')[0])
            imageDfDict["imageId"].append(index)

        imageIdDf = pd.DataFrame.from_dict(imageDfDict)
        imageIdDf.to_csv(IMAGE_IDS_CSV)
        return imageIdDf

    def getImageIdsFromFile(self):
        if not os.path.exists(IMAGE_IDS_CSV):
            return self.createImageIdsFile()

        return pd.read_csv(IMAGE_IDS_CSV)

    def getImageName(self, imageId):
        return self.imageIdToNameDf.loc[imageId, 'imageName']

    def getImageId(self, imageName):
        return int(self.imageNameToIdDf.loc[imageName, 'imageId'])

    def getImageIds(self, imagePaths):
        return [self.getImageId(self.imageHelper.getImageName(imagePath)) for imagePath in imagePaths]

    def getImagePaths(self, imageIds):
        return [self.imageHelper.getImagePath(self.getImageName(imageId)) for imageId in imageIds]
