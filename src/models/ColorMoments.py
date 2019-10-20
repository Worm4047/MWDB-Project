import numpy as np
import cv2
import math
from scipy.stats import skew
from src.models.interfaces.Model import Model
import time

class ColorMoments(Model):
    def __init__(self, image, verticalWindows, horizontalWindows):
        self.validateImage(image)
        super(ColorMoments, self).__init__()
        self.BLOCK_SIZE = min(verticalWindows, horizontalWindows)
        self.widowHeight = verticalWindows
        self.widowWidth = horizontalWindows
        self.imageHeight, self.imageWidth, _ = image.shape
        featureVectorHeight = math.ceil(self.imageHeight / self.widowHeight)
        featureVectorWidth = math.ceil(self.imageWidth / self.widowWidth)

        # Mean for all three channels
        # Initialised to zeros
        self.meanFeatureVector = np.zeros((featureVectorHeight, featureVectorWidth, 3))
        # Variance for all three channels
        self.varianceFeatureVector = np.zeros((featureVectorHeight, featureVectorWidth, 3))
        # Skew for all three channels
        self.skewFeatureVector = np.zeros((featureVectorHeight, featureVectorWidth, 3))

        # Populates above feature vectors
        self.computeColorMoments(image)

        # Concatinated feature vector
        self.featureVector = np.concatenate((self.meanFeatureVector, self.varianceFeatureVector, self.skewFeatureVector), axis=2)

    def validateImage(self, image):
        if image.shape is None:
            raise ValueError("Not a np array")
        if len(image.shape) != 3:
            raise ValueError("Invalid Image")
        if image.shape[2] != 3:
            raise ValueError("Invalid Image")

    def getFeatures(self):
        return self.featureVector.flatten()

    def getFeaturesWithDim(self):
        return self.featureVector

    def compare(self, colorModel):
        if not isinstance(colorModel, Model):
            raise ValueError("Not a instance of ColorModel")

        return np.linalg.norm(self.featureVector.flatten() - colorModel.getFeatures())

    def getSkew(self, block, mean):
        blockFlat = block.flatten()
        sumOfL3Diffs = 0
        for value in blockFlat:
            sumOfL3Diffs += math.pow(value - mean, 3)

        return np.power(sumOfL3Diffs/blockFlat.shape[0], 1/3)

    # Computes color moments for the give image
    # Breaks the image into multiple blocks an
    def computeColorMoments(self, image):
        channelY, channelU, channelV = cv2.split(image)

        i, j = 0, 0
        while i < self.imageHeight:
            j = 0

            while j < self.imageWidth:
                blockY = channelY[i: i + self.BLOCK_SIZE, j: j + self.BLOCK_SIZE]
                blockU = channelU[i: i + self.BLOCK_SIZE, j: j + self.BLOCK_SIZE]
                blockV = channelV[i: i + self.BLOCK_SIZE, j: j + self.BLOCK_SIZE]

                meanStart = time.time()
                meanY = np.mean(blockY)
                meanU = np.mean(blockU)
                meanV = np.mean(blockV)
                self.meanFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 0] = meanY
                self.meanFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 1] = meanU
                self.meanFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 2] = meanV
                # print("Mean calculation time: {} seconds".format(time.time() - meanStart))

                varianceStart = time.time()
                self.varianceFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 0] = np.std(blockY)
                self.varianceFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 1] = np.std(blockU)
                self.varianceFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 2] = np.std(blockV)
                # print("Variance calculation time: {} seconds".format(time.time() - varianceStart))

                skewStart = time.time()
                self.skewFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 0] = skew(blockY, axis=None)
                self.skewFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 1] = skew(blockU, axis=None)
                self.skewFeatureVector[math.floor(i / self.BLOCK_SIZE), math.floor(j / self.BLOCK_SIZE), 2] = skew(blockV, axis=None)
                # print("Skew calculation time: {} seconds".format(time.time() - skewStart))

                self.momentsY = np.concatenate((self.meanFeatureVector[:, :, 0], self.varianceFeatureVector[:, :, 0],
                                                self.skewFeatureVector[:, :, 0]))
                j += self.BLOCK_SIZE
            i += self.BLOCK_SIZE
