from src.models.enums.models import ModelType
from src.distance.distanceCalulator import DistanceType
from src.constants import GRAPH_STORE, DATABASE_PATH
from src.distance.distanceArchiver import DistanceArchiver
from src.common.imageIndex import ImageIndex
from src.common.imageHelper import ImageHelper
from enum import Enum
from scipy import sparse
import time
import os
import numpy as np
import glob

class GraphType(Enum):
    TRANSITION=1
    WEIGHTED_UNNORMALISED=2
    UNWEIGHTED=3

class GraphArchiver:
    modelType = None
    distanceType = None
    graphFolderName = None
    distanceArchiver = None
    imageIndex = None
    imageHelper = None
    imagesCount = None

    def __init__(self, k, modelType=ModelType.CM, distanceType=DistanceType.EUCLIDEAN):
        self.modelType = modelType
        self.distanceType = distanceType
        self.k = k
        self.graphFolderName = self.getGraphFolderName()
        self.distanceArchiver = DistanceArchiver()
        self.imageIndex = ImageIndex()
        self.imageHelper = ImageHelper()
        self.setImagesCount()
        self.checkGraphDir()
        self.createGraph()

    def setImagesCount(self):
        self.imagesCount = len(glob.glob(os.path.join(DATABASE_PATH, "*.jpg")))

    def checkGraphDir(self):
        if not os.path.exists(self.graphFolderName):
            os.makedirs(self.graphFolderName)

    def createGraph(self):
        if os.path.exists(self.getGraphFilePath(GraphType.UNWEIGHTED)): return

        distance_matrix = self.distanceArchiver.getDistances().to_numpy()
        distance_matrix = distance_matrix[:, 1:]
        distanceMatrixLen = len(distance_matrix)

        for row_index, row in enumerate(distance_matrix):
            adjacent_items = [arg for arg in np.argsort(row)[1:self.k + 1]]
            newRow = [0 for j in range(0, distanceMatrixLen)]
            for adjacent_item in adjacent_items:
                newRow[adjacent_item] = row[adjacent_item]

            distance_matrix[row_index] = newRow

        self.saveGraph(distance_matrix, GraphType.WEIGHTED_UNNORMALISED)
        self.normaliseEdgesForNodes(distance_matrix)
        self.testNormalisation(distance_matrix)
        self.saveGraph(distance_matrix, GraphType.TRANSITION)
        distance_matrix[distance_matrix > 0] = 1
        self.saveGraph(distance_matrix, GraphType.UNWEIGHTED)

    def normaliseEdgesForNodes(self, distanceMatrix):
        for index, row in enumerate(distanceMatrix):
            newRow = row/np.sum(row)
            distanceMatrix[index] = newRow
            distanceMatrix[:, index] = newRow

    def testNormalisation(self, matrix):
        for index in range(0, len(matrix)):
            if not np.sum(matrix[:, index]) != 1:
                print("Bug in col: {} | val: {}".format(index, np.sum(matrix[:, index])))

            if not np.sum(matrix[index]) != 1:
                print("Bug in row: {} | val: {}".format(index, np.sum(matrix[index])))

    def getGraphFolderName(self):
        return os.path.join(GRAPH_STORE, "{}_{}_{}".format(self.modelType.name, self.distanceType.name, self.k))

    def saveGraph(self, matrix, graphType):
        '''
        :param matrix: Dense matrix representing the graph
        :param graphType: of type GraphType
        :return: void

        Converts the dense matrix into a sparse matrix and saves it
        '''
        sparse.save_npz(self.getGraphFilePath(graphType), sparse.csc_matrix(matrix))

    def getGraph(self, graphType):
        '''
        :param graphType: of type GraphType
        :return: spare matrix representing a grpah
        '''

        print("Loading the graph...")
        start = time.time()
        grah = sparse.load_npz(self.getGraphFilePath(graphType))
        print("Loaded the graph: {} | Time take: {}".format(self.getGraphFilePath(graphType), time.time() - start))
        return grah

    def getGraphFilePath(self, graphType):
        if not isinstance(graphType, GraphType):
            raise ValueError("graphType should be of type GraphType")

        return os.path.join(self.graphFolderName, "{}.npz".format(graphType.name))

    def getTeleportationVector(self, imageIds):
        randonJumpProb = 1/len(imageIds)

        teleportationVector = [0 for i in range(0, self.imagesCount)]
        for imageId in imageIds:
            teleportationVector[imageId] = randonJumpProb

        return np.array(teleportationVector).transpose()

    def getPersonalisedPageRankForThreeImages(self, imageIds, iter=None, thres=0.8):
        if iter == None and thres == None:
            raise ValueError("One of iter and thres should be passed")

        transitionMatrixInSparse = self.getGraph(GraphType.TRANSITION)
        pageRank = self.getTeleportationVector(imageIds)

        if iter is not None:
            return self.runPPRWithTransitionMatrixForIter(pageRank, iter, transitionMatrixInSparse)
        elif thres is not None:
            return self.runPPRWithTransitionMatrixForThres(pageRank, thres, transitionMatrixInSparse)

    def runPPRWithTransitionMatrixForThres(self, pageRank, thres, transitionMatrixInSparse):
        iterCount = 0
        while (True):
            prevPageRank = pageRank
            pageRank = transitionMatrixInSparse @ pageRank

            error = np.linalg.norm(pageRank - prevPageRank)
            if error < thres: break

            if iterCount % 1 == 0: print("Iteration: {} | Error: {}".format(iterCount, error))
            iterCount += 1

        return pageRank

    def runPPRWithTransitionMatrixForIter(self, pageRank, iter, transitionMatrixInSparse):
        for iterCount in range(0, iter):
            prevPageRank = pageRank
            pageRank = transitionMatrixInSparse @ pageRank

            error = np.linalg.norm(pageRank - prevPageRank)
            print("Iteration: {} | Error: {}".format(iterCount, error))

        return pageRank

    def getSimilarImageIdsFromGraph(self, imageIds, K):
        '''
        :param imageIds: List of imageIds of 3 query images. List can be of any length.
        :param K: The number of similar images to be extracted
        :return: imageIds of K similar images
        '''

        pageRank = self.getPersonalisedPageRankForThreeImages(imageIds, thres=1e-50)
        return np.flip(np.argsort(pageRank))[0:K]













