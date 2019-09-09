import numpy as np
import cv2
import math

class ColorMoments:
    def __init__(self, image, verticalWindows, horizontalWindows):
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

    # Computes color moments for the give image
    # Breaks the image into multiple blocks an
    def computeColorMoments(self, image):
        channelY, channelU, channelV = cv2.split(image)

        i, j = 0, 0
        while i < self.imageHeight:
            j = 0
            while j < self.imageWidth:
                blockY = channelY[i: i + 100, j: j + 100]
                blockU = channelU[i: i + 100, j: j + 100]
                blockV = channelV[i: i + 100, j: j + 100]

                self.meanFeatureVector[math.floor(i / 100), math.floor(j / 100), 0] = np.mean(blockY)
                self.meanFeatureVector[math.floor(i / 100), math.floor(j / 100), 1] = np.mean(blockU)
                self.meanFeatureVector[math.floor(i / 100), math.floor(j / 100), 2] = np.mean(blockV)

                self.varianceFeatureVector[math.floor(i / 100), math.floor(j / 100), 0] = np.std(blockY)
                self.varianceFeatureVector[math.floor(i / 100), math.floor(j / 100), 1] = np.std(blockU)
                self.varianceFeatureVector[math.floor(i / 100), math.floor(j / 100), 2] = np.std(blockV)

                self.skewFeatureVector[math.floor(i / 100), math.floor(j / 100), 0] = 0
                self.skewFeatureVector[math.floor(i / 100), math.floor(j / 100), 1] = 0
                self.skewFeatureVector[math.floor(i / 100), math.floor(j / 100), 2] = 0

                j += 100
            i += 100
