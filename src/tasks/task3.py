from src.common.imageHelper import ImageHelper
from src.common.imageIndexNoCache import ImageIndexNoCache
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

        imagePaths = glob.glob(os.path.join(imageDir, "*.jpg"))
        imageClasses = [ImageClass.NONE for _ in imagePaths]
        self.imageIndex = ImageIndexNoCache(imagePaths, imageClasses)
        self.graphArchiver = GraphArchiverNoCache(k, imagePaths=imagePaths, imageClasses=imageClasses, modelTypes=modelTypes)

    def getSimilarImagePaths(self, K, queryImagePaths):
        queryImageIds = self.imageIndex.getImageIdsForPaths(queryImagePaths)
        imageIds = self.graphArchiver.getSimilarImageIdsFromGraph(queryImageIds, K)

        return self.imageIndex.getImagePathsForImageIds(imageIds)

    def visualiseSimilarImages(self, K, queryImagePaths):
        imagePaths = self.getSimilarImagePaths(K, queryImagePaths)
        imageDict = {}
        for imagePath in imagePaths:
            imageDict[self.imageHelper.getImageName(imagePath)] = self.imageHelper.getRGBImage(imagePath)

        plotFigures(imageDict)

