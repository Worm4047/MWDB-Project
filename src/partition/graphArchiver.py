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
    modelTypes = []
    distanceType = None
    graphFolderName = None
    distanceArchivers = []
    imageIndex = None
    imageHelper = None
    imagesCount = None
    beta = 0.15

    def __init__(self, k, modelTypes=None, distanceType=DistanceType.EUCLIDEAN):
        if modelTypes is None:
            modelTypes = [ModelType.CM]

        self.modelTypes = modelTypes
        self.distanceType = distanceType
        self.k = k
        self.graphFolderName = self.getGraphFolderName()
        for modelType in modelTypes:
            self.distanceArchivers.append(DistanceArchiver(modelType=modelType))
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

    def getDistanceMatrix(self):
        distanceMatrix = None
        for distanceArchiver in self.distanceArchivers:
            if distanceMatrix is None:
                distanceMatrix = distanceArchiver.getDistances().to_numpy()
            else:
                distanceMatrix += distanceArchiver.getDistances().to_numpy()

        return distanceMatrix

    def createGraph(self):
        if os.path.exists(self.getGraphFilePath(GraphType.UNWEIGHTED)): return

        distance_matrix = self.getDistanceMatrix()
        distance_matrix = distance_matrix[:, 1:]
        distanceMatrixLen = len(distance_matrix)

        similarityMatrix = np.exp(-distance_matrix)

        for colIndex in range(0, self.imagesCount):
            col = similarityMatrix[:, colIndex]
            adjacent_items = [arg for arg in np.flip(np.argsort(col))[1:self.k + 1]]
            newCol = [0 for j in range(0, distanceMatrixLen)]
            for adjacent_item in adjacent_items:
                newCol[adjacent_item] = col[adjacent_item]

            similarityMatrix[:, colIndex] = newCol

        self.saveGraph(similarityMatrix, GraphType.WEIGHTED_UNNORMALISED)
        self.normaliseEdgesForNodes(similarityMatrix)
        # self.testNormalisation(distance_matrix)
        self.saveGraph(similarityMatrix, GraphType.TRANSITION)
        similarityMatrix[similarityMatrix > 0] = 1
        self.saveGraph(similarityMatrix, GraphType.UNWEIGHTED)

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

    def getGraphFolderName(self):
        return os.path.join(GRAPH_STORE, "{}_{}_{}".format(",".join([x.name for x in self.modelTypes]), self.distanceType.name, self.k))

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

        pageRank = self.getPersonalisedPageRankForThreeImages(imageIds, thres=1e-05)
        return np.flip(np.argsort(pageRank))[0:K + 3]

    def set_seed_values(self, rank_array, seed_values, damping_factor):
        for seed_index in seed_values:
            rank_array.itemset(seed_index, (1 - damping_factor) / 3)

    def get_personalized_page_rank_general(self, sim_graph, imageIds, tolerance=1.0e-5):
        return self.get_personalized_page_rank_without_transition_matrix(sim_graph, imageIds, tolerance)

    def get_personalized_page_rank_without_transition_matrix(self, sim_graph, seed_values, tolerance=1.0e-5):
        np_graph = np.array(sim_graph)
        graph_transpose = np_graph.transpose()
        damping_factor = 0.85
        total_nodes = len(sim_graph)
        error = 1
        iteration = 0
        # page_rank_current = np.full(shape=(total_nodes, total_nodes), fill_value=(1 - damping_factor) / total_nodes)
        # initial_page_rank = np.full(shape=(total_nodes, total_nodes), fill_value=(1 - damping_factor) / total_nodes)

        page_rank_current = np.array([0.0 for i in range(0, total_nodes)])
        initial_page_rank = np.array([0.0 for i in range(0, total_nodes)])
        self.set_seed_values(page_rank_current, seed_values, damping_factor)
        self.set_seed_values(initial_page_rank, seed_values, damping_factor)

        while error > tolerance:
            index = 0
            print("Iteration: {} | Tolerance: {}".format(iteration, error))
            for row in graph_transpose:
                edge_indexes = np.nonzero(row > -1)[0]
                for edge_index in edge_indexes:
                    page_rank_current[index] += initial_page_rank[edge_index] * damping_factor / self.k
                index += 1
            # page_rank_sum = np.sum(page_rank_current)
            # page_rank_current = page_rank_current/page_rank_sum
            error = np.linalg.norm(page_rank_current - initial_page_rank, 2)
            initial_page_rank = page_rank_current
            page_rank_current = np.array([0.0 for i in range(0, total_nodes)])
            self.set_seed_values(page_rank_current, seed_values, damping_factor)
            iteration += 1

        print("Iteration: {}".format(iteration))

        return initial_page_rank














