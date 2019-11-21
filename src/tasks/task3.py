from src.common.imageHelper import ImageHelper
from src.common.imageIndex import ImageIndex
from src.partition.graphArchiver import GraphArchiver

class Task3:
    imageHelper = None
    imageIndex = None
    graphArchiver = None

    def __init__(self, k):
        self.imageHelper = ImageHelper()
        self.imageIndex = ImageIndex()
        self.graphArchiver = GraphArchiver(k)

    def getSimilarImagePaths(self, K, queryImagePaths):
        queryImageIds = self.imageIndex.getImageIds(queryImagePaths)
        imageIds = self.graphArchiver.getSimilarImageIdsFromGraph(queryImageIds, K)

        print(self.imageIndex.getImagePaths(imageIds))
