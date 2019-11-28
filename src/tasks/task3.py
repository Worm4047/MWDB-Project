from src.common.imageHelper import ImageHelper
from src.common.imageIndex import ImageIndex
from src.partition.graphArchiver import GraphArchiver
from src.archive.plotHelper import plotFigures
from src.models.enums.models import ModelType
import numpy as np

class Task3:
    imageHelper = None
    imageIndex = None
    graphArchiver = None

    def __init__(self, k, modelTypes=None):
        if modelTypes is None:
            modelTypes = [ModelType.CM, ModelType.HOG]
        self.imageHelper = ImageHelper()
        self.imageIndex = ImageIndex()
        self.graphArchiver = GraphArchiver(k, modelTypes=modelTypes)

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

