from src.common.helper import getImagePathsFromDB
from src.models.enums.models import ModelType
from src.distance.distanceCalulator import DistanceCalculator
from src.common.imageHelper import ImageHelper
from src.common.imageIndex import ImageIndex
from src.constants import FEATURE_STORE
import pandas as pd
import cv2
import os
from src.distance.distanceCalulator import DistanceType
import time
import numpy as np

class FeatureArchiver:
    imageHelper = None
    distanceCalculator = None
    imageIndex = None

    def __init__(self, modelType=ModelType.CM):
        self.imageHelper = ImageHelper()
        self.imageIndex = ImageIndex()
        self.checkFeaturesStoreDir()
        self.createFeatureVectors(modelType)

    def checkFeaturesStoreDir(self):
        if not os.path.isdir(FEATURE_STORE):
            os.makedirs(FEATURE_STORE)

    def createFeatureVectors(self, modelType):
        imagePaths = getImagePathsFromDB()
        imagesCount = len(imagePaths)
        featureCalculationStart = time.time()
        for index, imagePath in enumerate(imagePaths):
            imageFeatureStart = time.time()
            if os.path.exists(self.getFeaturesFilePathForImage(imagePath, modelType)): continue

            featureVector = self.imageHelper.getImageFeatures(imagePath, modelType, retriveshape=True)
            self.saveFeaturesForImage(imagePath, featureVector, modelType)

            print("Feature calculation | Processed {} out of {} | Time taken for last image: {}"
                    .format(index, imagesCount, time.time() - imageFeatureStart))

        print("Feature calculation | Total time taken : {}".format(time.time() - featureCalculationStart))

    def saveFeaturesForImage(self, imagePath, featureVector, modelType):
        np.save(self.getFeaturesFilePathForImage(imagePath, modelType), featureVector)

    def createAndStoreImageFeatures(self, imagePath, modelType):
        featureVector = self.imageHelper.getImageFeatures(imagePath, modelType, retriveshape=True)
        self.saveFeaturesForImage(imagePath, featureVector, modelType)

        return featureVector

    def getFeaturesForImage(self, imagePath, modelType):
        featuresFilePath = self.getFeaturesFilePathForImage(imagePath, modelType)
        if os.path.exists(featuresFilePath):
            return np.load(featuresFilePath)
        else:
            return self.createAndStoreImageFeatures(imagePath, modelType)

    def getFeaturesFilePathForImage(self, imagePath, modelType):
        return os.path.join(FEATURE_STORE, "{}_{}.npy".format(self.imageHelper.getImageName(imagePath), modelType.name))

