from src.common.helper import getImagePathsFromDB
from src.models.enums.models import ModelType
from src.distance.distanceCalulator import DistanceCalculator
from src.common.imageHelper import ImageHelper
from src.common.imageIndex import ImageIndex
from src.constants import DISTANCES_PAIR_WISE_STORE, DISTANCES_MATRIX_STORE
from src.distance.distanceCalulator import DistanceType
from src.models.featureArchiver import FeatureArchiver
from src.dimReduction.dimRedHelper import DimRedHelper
import pandas as pd
import cv2
import os
from scipy import spatial

import time

class DistanceArchiver:
    imageHelper = None
    distanceCalculator = None
    imageIndex = None
    featureArchiver = None
    dimRedHelper = None
    modelType = None
    distanceType = None

    def __init__(self, modelType=ModelType.CM, distanceType=DistanceType.EUCLIDEAN, checkArchive=False):
        self.imageHelper = ImageHelper()
        self.distanceCalculator = DistanceCalculator()
        self.imageIndex = ImageIndex()
        self.featureArchiver = FeatureArchiver(modelType=modelType)
        self.imageHelper = ImageHelper()
        self.dimRedHelper = DimRedHelper()
        self.modelType = modelType
        self.distanceType = distanceType
        if checkArchive:
            self.checkDistancesStoreDir()
            self.createPairWiseDistances()

    def checkDistancesStoreDir(self):
        if not os.path.isdir(DISTANCES_MATRIX_STORE):
            os.makedirs(DISTANCES_MATRIX_STORE)

    def createPairWiseDistances(self):
        if os.path.exists(self.getDistancesFilename()): return

        imagePaths = getImagePathsFromDB()
        dataMatrix = self.dimRedHelper.getDataMatrix(imagePaths, self.modelType)
        imageNames = [self.imageHelper.getImageName(x) for x in imagePaths]
        imageIds = [self.imageIndex.getImageId(x) for x in imageNames]

        start = time.time()
        print("Calculating pair wise distances...")
        distancesMatrix = spatial.distance.cdist(dataMatrix, dataMatrix, metric='euclidean')
        print("Pair wise distance calculation complete | Time taken: {}".format(time.time() - start))

        distancesDict = {}
        for index, imageId in enumerate(imageIds):
            distancesDict[imageId] = distancesMatrix[:, imageId]

        distancesDf = pd.DataFrame.from_dict(distancesDict)

        print("Storing pair wise distances...")
        storingStart = time.time()
        self.savePairWiseDistances(distancesDf)
        print("Pair wise distances stored | Time taken: {}".format(time.time() - storingStart))

    def savePairWiseDistances(self, distancesDf):
        distancesDf.to_csv(self.getDistancesFilename())

    def getDistancesFilename(self):
        return os.path.join(DISTANCES_MATRIX_STORE, "{}_{}.csv".format(self.modelType.name, self.distanceType.name))

    def getDistances(self):
        imageDistanceFilePath = self.getDistancesFilename()
        if os.path.exists(imageDistanceFilePath):
            print("Loading distances matrix for modelType: {} and distanceType: {}".format(self.modelType.name,
                                                                                           self.distanceType.name))
            loadingStart = time.time()
            distancesDf = pd.read_csv(imageDistanceFilePath)
            print("Distances matrix loaded | Time taken: {}".format(time.time() - loadingStart))
            return distancesDf
        else:
            raise ValueError("No distances file exists for model: {} and distance type: {}"
                             .format(self.modelType.name, self.distanceType.name))
