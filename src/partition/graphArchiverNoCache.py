from src.models.enums.models import ModelType
from src.distance.distanceCalulator import DistanceType
from src.constants import GRAPH_STORE, DATABASE_PATH, DORSAL_DATABASE_PATH, PALMAR_DATABASE_PATH, DORSAL_GRAPH_STORE, PALMAR_GRAPH_STORE
from src.distance.distanceArchiver import DistanceArchiver
from src.distance.distanceArchiverNoCache import DistanceArchiverNoCache
from src.common.imageIndex import ImageIndex
from src.common.imageHelper import ImageHelper
from src.common.imageIndexNoCache import ImageIndexNoCache
from src.partition.graphArchiver import GraphType
from src.classifiers.pprClassifier import ImageClass
from enum import Enum
from scipy import sparse
import shutil
import time
import os
import numpy as np
import glob

class GraphArchiverNoCache:
    modelTypes = []
    distanceType = None
    distanceArchivers = []
    imageClass = None
    imageIndex = None
    imagesCount = None
    imageHelper = None
    beta = 0.15
    similarityMatrixInSparse = None
    tansitionMatrixInSparse = None

    def __init__(self, k, imagePaths, imageClasses, modelTypes=None, distanceType=DistanceType.EUCLIDEAN):
        if modelTypes is None:
            modelTypes = [ModelType.CM]

        self.imagePaths = imagePaths
        self.imagesCount = len(self.imagePaths)
        self.imageClasses = imageClasses
        self.imageIndex = ImageIndexNoCache(self.imagePaths, self.imageClasses)

        self.modelTypes = modelTypes
        self.distanceType = distanceType
        self.k = k
        self.distanceArchivers = []
        for modelType in modelTypes:
            self.distanceArchivers.append(DistanceArchiverNoCache(modelType=modelType, imagePaths=imagePaths))

        self.imageHelper = ImageHelper()
        self.createGraph()

    def getImagesCount(self):
        return len(self.imagePaths)

    def getImagePaths(self):
        return self.imagePaths

    def getDistanceMatrix(self):
        distanceMatrix = None
        for distanceArchiver in self.distanceArchivers:
            if distanceMatrix is None:
                distanceMatrix = distanceArchiver.getPairWiseDistances()
            else:
                distanceMatrix += distanceArchiver.getPairWiseDistances()

        return distanceMatrix

    def createGraph(self):
        distance_matrix = self.getDistanceMatrix()
        distanceMatrixLen = len(distance_matrix)
        similarityMatrix = np.exp(-distance_matrix)

        for colIndex in range(0, self.imagesCount):
            col = similarityMatrix[:, colIndex]
            adjacent_items = [arg for arg in np.flip(np.argsort(col))[1:self.k + 1]]
            newCol = [0 for j in range(0, distanceMatrixLen)]
            for adjacent_item in adjacent_items:
                newCol[adjacent_item] = col[adjacent_item]

            similarityMatrix[:, colIndex] = newCol

        self.similarityMatrixInSparse = sparse.csc_matrix(similarityMatrix)
        self.normaliseEdgesForNodes(similarityMatrix)
        self.tansitionMatrixInSparse = sparse.csc_matrix(similarityMatrix)

    def normaliseEdgesForNodes(self, matrix):
        for index in range(0, self.imagesCount):
            col = matrix[:, index]
            if np.sum(col) == 0:
                print("Boom | index: {}".format(index))

            newCol = col/np.sum(col)
            matrix[:, index] = newCol

    def testNormalisation(self, matrix):
        for index in range(0, len(matrix)):
            if not np.sum(matrix[:, index]) != 1:
                print("Bug in col: {} | val: {}".format(index, np.sum(matrix[:, index])))

            if not np.sum(matrix[index]) != 1:
                print("Bug in row: {} | val: {}".format(index, np.sum(matrix[index])))

    def getImageClassForImageId(self, imageId):
        return self.imageIndex.getImageClassForImageId(imageId)

    def getGraph(self, graphType):
        '''
        :param graphType: of type GraphType
        :return: spare matrix representing a grpah
        '''

        if graphType == GraphType.TRANSITION:
            return self.tansitionMatrixInSparse

        return None

    def getTeleportationVector(self, imageIds):
        randonJumpProb = 1/len(imageIds)

        teleportationVector = [0 for i in range(0, self.imagesCount)]
        for imageId in imageIds:
            teleportationVector[imageId] = randonJumpProb

        return np.array(teleportationVector).transpose()

    def getPersonalisedPageRankForImages(self, imageIds, iter=None, thres=1e-05):
        if iter == None and thres == None:
            raise ValueError("One of iter and thres should be passed")

        transitionMatrixInSparse = self.getGraph(GraphType.TRANSITION)
        teleportationVector = self.getTeleportationVector(imageIds)
        initialPageRank = [1/self.imagesCount for j in range(0, self.imagesCount)]

        initialPageRank = teleportationVector

        # return self.get_personalized_page_rank_general(transitionMatrixInSparse.todense(), imageIds)

        if iter is not None:
            return self.runPPRWithTransitionMatrixForIter(initialPageRank, iter, transitionMatrixInSparse, teleportationVector)
        elif thres is not None:
            return self.runPPRWithTransitionMatrixForThres(initialPageRank, thres, transitionMatrixInSparse, teleportationVector)

    def runPPRWithTransitionMatrixForThres(self, pageRank, thres, transitionMatrixInSparse, teleportationVector):
        iterCount = 0
        randomJumpProb = teleportationVector * self.beta
        while (True):
            prevPageRank = pageRank
            jumpFromNeighboursPrb = (1 - self.beta) * (transitionMatrixInSparse @ pageRank)
            pageRank = jumpFromNeighboursPrb + randomJumpProb

            error = np.linalg.norm(pageRank - prevPageRank)
            if error < thres: break

            if iterCount % 500 == 0: print("Iteration: {} | Error: {}".format(iterCount, error))
            iterCount += 1

        return pageRank

    def getPPR(self, thres=1e-05):
        iterCount = 0
        transitionMatrixInSparse = self.getGraph(GraphType.TRANSITION)
        pageRank = [1 / self.imagesCount for j in range(0, self.imagesCount)]
        error = -1

        while (True):
            prevPageRank = pageRank
            pageRank = transitionMatrixInSparse @ pageRank

            error = np.linalg.norm(pageRank - prevPageRank)
            if error < thres: break

            if iterCount % 500 == 0: print("Iteration: {} | Error: {}".format(iterCount, error))
            iterCount += 1

        print("Total iteration: {} | Error: {}".format(iterCount, error))
        return pageRank

    def runPPRWithTransitionMatrixForIter(self, pageRank, iter, transitionMatrixInSparse, teleportationVector):
        randomJumpProb = teleportationVector * self.beta
        for iterCount in range(0, iter):
            prevPageRank = pageRank
            jumpFromNeighboursPrb = (1-self.beta) * (transitionMatrixInSparse @ pageRank)
            pageRank = jumpFromNeighboursPrb + randomJumpProb

            error = np.linalg.norm(pageRank - prevPageRank)
            if iterCount % 500 == 0: print("Iteration: {} | Error: {}".format(iterCount, error))

        return pageRank

    def getSimilarImageIdsFromGraph(self, imageIds, K):
        '''
        :param imageIds: List of imageIds of 3 query images. List can be of any length.
        :param K: The number of similar images to be extracted
        :return: imageIds of K similar images
        '''

        pageRank = self.getPersonalisedPageRankForImages(imageIds, thres=1e-05)
        return np.flip(np.argsort(pageRank))[0:K]
