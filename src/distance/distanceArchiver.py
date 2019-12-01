from src.common.helper import getImagePathsFromDB
from src.models.enums.models import ModelType
from src.distance.distanceCalulator import DistanceCalculator
from src.common.imageHelper import ImageHelper
from src.common.imageIndex import ImageIndex
from src.constants import DISTANCES_MATRIX_STORE, PALMAR_DISTANCES_MATRIX_STORE, DORSAL_DISTANCES_MATRIX_STORE
from src.distance.distanceCalulator import DistanceType
from src.models.featureArchiver import FeatureArchiver
from src.dimReduction.dimRedHelper import DimRedHelper
from src.classifiers.pprClassifier import ImageClass
from src.constants import DORSAL_DATABASE_PATH, PALMAR_DATABASE_PATH, DATABASE_PATH
import pandas as pd
import cv2
import os
from scipy import spatial
import time
import glob
import shutil

class DistanceArchiver:
    imageHelper = None
    distanceCalculator = None
    imageIndex = None
    featureArchiver = None
    dimRedHelper = None
    modelType = None
    distanceType = None
    databasePath = None
    imageClass = None
    distanceMatrixStore = None

    def __init__(self, modelType=ModelType.CM, distanceType=DistanceType.EUCLIDEAN, imageClass=None):
        self.imageClass = imageClass
        self.setDatabasePath()
        self.setDistanceMatrixStore()

        self.imageHelper = ImageHelper()
        self.distanceCalculator = DistanceCalculator()
        self.imageIndex = ImageIndex(imageClass)
        self.featureArchiver = FeatureArchiver(modelType=modelType)
        self.imageHelper = ImageHelper()
        self.dimRedHelper = DimRedHelper()
        self.modelType = modelType
        self.distanceType = distanceType
        self.checkDistancesStoreDir()
        self.createPairWiseDistances()

    def setDatabasePath(self):
        if self.imageClass == ImageClass.DORSAL:
            self.databasePath = DORSAL_DATABASE_PATH
        if self.imageClass == ImageClass.PALMAR:
            self.databasePath = PALMAR_DATABASE_PATH
        if self.imageClass is None:
            self.databasePath = DATABASE_PATH

    def setDistanceMatrixStore(self):
        if self.imageClass == ImageClass.DORSAL:
            self.distanceMatrixStore = DORSAL_DISTANCES_MATRIX_STORE
        if self.imageClass == ImageClass.PALMAR:
            self.distanceMatrixStore = PALMAR_DISTANCES_MATRIX_STORE
        if self.imageClass is None:
            self.distanceMatrixStore = DISTANCES_MATRIX_STORE

    def checkDistancesStoreDir(self):
        if not os.path.isdir(self.distanceMatrixStore):
            os.makedirs(self.distanceMatrixStore)

    def getImagePathsFromDB(self):
        return glob.glob(os.path.join(self.databasePath, "*.jpg"))

    def createPairWiseDistances(self):
        if os.path.exists(self.getDistancesFilename()): return

        imagePaths = self.getImagePathsFromDB()
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
        return os.path.join(self.distanceMatrixStore, "{}_{}.csv".format(self.modelType.name, self.distanceType.name))

    def deleteDistanceMatrix(self):
        if os.path.exists(self.distanceMatrixStore):
            shutil.rmtree(self.distanceMatrixStore)

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
