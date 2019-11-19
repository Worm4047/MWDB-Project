import os
import pandas as pd
from src.constants import DATABASE_PATH, IMAGE_IDS_CSV
import glob
import numpy as np

class DataFrameParser:
    imageNameToIdDf = None
    imageIdToNameDf = None

    def __init__(self):
        self.imageNameToIdDf = self.getImageIdsFromFile()
        self.imageIdtoNameDf = self.imageNameToIdDf.copy()

        self.imageNameToIdDf.set_index("imageName", inplace=True)
        self.imageIdtoNameDf.set_index("imageId", inplace=True)

    def createImageIdsFile(self):
        imagePaths = glob.glob(os.path.join(DATABASE_PATH, ".jpg"))
        imageDfDict = {
            "imageName": [],
            "imageId": []
        }
        for index, imagePath in enumerate(imagePaths):
            imageDfDict["imageName"].add(os.path.basename(imagePath).split('.')[0])
            imageDfDict["imageId"].add(index)

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
        return self.imageNameToIdDf.loc(imageName, 'imageId')

if __name__ == "__main__":
    imageIds = DataFrameParser()
    print(imageIds.getImageId("Hand_0000002.jpg"))
    print(imageIds.getImageName(imageIds.getImageId("Hand_0000002.jpg")))
