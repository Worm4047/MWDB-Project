from src.partition.graphArchiver import GraphArchiver
from src.common.imageIndex import ImageIndex

class Task6PPR:
    graphArchiver = None

    def __init__(self):
        self.graphArchiver = GraphArchiver(10)
        self.imageIndex = ImageIndex()

    def getRelaventImages(self, userFeedback):
        relaventImages = userFeedback['relevant']
        irrelaventImages = userFeedback['nonrelevant']

        return self.imageIndex.getImagePaths(self.graphArchiver.getSimilarImageIdsFromGraph(self.imageIndex.getImageIds(relaventImages),
                                                       len(relaventImages) + len(irrelaventImages)))


