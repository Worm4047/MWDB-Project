from src.common.imageHelper import ImageHelper
from src.common.imageIndex import ImageIndex
from src.partition.graphArchiverNoCache import GraphArchiverNoCache
from src.archive.plotHelper import plotFigures
from src.models.enums.models import ModelType
from src.classifiers.pprClassifier import ImageClass
import numpy as np
import os
import glob

class Task3:
    imageHelper = None
    imageIndex = None
    graphArchiver = None

    def __init__(self, k, imageDir, modelTypes=None):
        if modelTypes is None:
            modelTypes = [ModelType.CM, ModelType.HOG]
        self.imageHelper = ImageHelper()
        self.imageIndex = ImageIndex()
        imagePaths = glob.glob(os.path.join(imageDir, "*.jpg"))
        imageClasses = [ImageClass.NONE for _ in imagePaths]
        self.graphArchiver = GraphArchiverNoCache(k, imagePaths=imagePaths, imageClasses=imageClasses, modelTypes=modelTypes)

    def getSimilarImagePaths(self, K, queryImagePaths):
        queryImageIds = self.imageIndex.getImageIds(queryImagePaths)
        imageIds = self.graphArchiver.getSimilarImageIdsFromGraph(queryImageIds, K)

        return self.imageIndex.getImagePaths(imageIds)

    def visualiseSimilarImages(self, K, queryImagePaths):
        imagePaths = self.getSimilarImagePaths(K, queryImagePaths)
        imageDict = {}
        for imagePath in imagePaths:
            imageDict[self.imageHelper.getImageName(imagePath)] = self.imageHelper.getRGBImage(imagePath)

        plotFigures(imageDict)

