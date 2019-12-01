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

class DistanceArchiverNoCache:
    imageHelper = None
    dimRedHelper = None
    modelType = None
    distancesMatrix = None
    imagePaths = None

    def __init__(self, imagePaths, modelType=ModelType.CM, distanceType=DistanceType.EUCLIDEAN):
        self.imageHelper = ImageHelper()
        self.featureArchiver = FeatureArchiver(modelType=modelType)
        self.imageHelper = ImageHelper()
        self.dimRedHelper = DimRedHelper()
        self.modelType = modelType
        self.distanceType = distanceType
        self.imagePaths = imagePaths

    def getPairWiseDistances(self):
        if self.distancesMatrix is not None: return self.distancesMatrix

        imagePaths = self.imagePaths
        dataMatrix = self.dimRedHelper.getDataMatrix(imagePaths, self.modelType)

        start = time.time()
        print("Calculating pair wise distances...")
        self.distancesMatrix = spatial.distance.cdist(dataMatrix, dataMatrix, metric='euclidean')
        print("Pair wise distance calculation complete | Time taken: {}".format(time.time() - start))

        return self.distancesMatrix
