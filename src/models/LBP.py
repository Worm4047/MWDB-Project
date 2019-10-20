from src.models.interfaces.Model import Model
from skimage.feature import local_binary_pattern
import numpy as np
import math

# Need to be considered blocksize
class LBP(Model):
    def __init__(self, grayScaleImage, blockSize, numPoints, radius, eps=1e-7):
        self.validateImage(grayScaleImage)
        self.numPoints = numPoints
        self.radius = radius
        self.eps = eps
        self.blockSize = blockSize
        super(LBP, self).__init__()
        self.computeFeatures(grayScaleImage)

    def getFeatures(self):
        return self.featureVector.flatten()

    def getFeatureWithDim(self):
        return self.featureVector

    def computeFeatures(self, img):
        feature_vector = []
        height = img.shape[0]
        width = img.shape[1]

        for y in range(0, height, self.blockSize):
            for x in range(0, width, self.blockSize):
                lbp_feature = self.getLBPFeatureForBlock(img[y:y + self.blockSize, x:x + self.blockSize])
                feature_vector.append(lbp_feature)

        self.featureVector = np.array(feature_vector)

    def getLBPFeatureForBlock(self, grayScaleImage):
        lbpFeatures = local_binary_pattern(grayScaleImage, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbpFeatures.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype(np.float)
        hist /= (hist.sum() + self.eps)
        return hist

    def validateImage(self, image):
        if image.shape is None:
            raise ValueError("Not a np array")
        if len(image.shape) != 2:
            raise ValueError("Invalid Image")

    def compare(self, lbpModel):
        if not isinstance(lbpModel, LBP):
            raise ValueError("Not a LBP model")

        return self.getSimilarityScoreLBP(self.featureVector, lbpModel.getFeatures())

    # Eucledian distance is calculated for 2 values val1 and val2
    def computeEucledian(self, val1, val2):
        # sum = 0
        # for v1, v2 in zip(val1, val2):
        #     sum += (v1 - v2) * (v1 - v2)
        # return math.sqrt(sum)
        return np.linalg.norm(val1 - val2)

    def getSimilarityScoreLBP(self, des1, des2):
        # Make both arrays of same length
        l1, l2 = des1.shape[0], des2.shape[0]
        diff = abs(l1 - l2)
        if l1 > l2:
            des2 = np.pad(des2, (0, diff), 'constant')
        else:
            des1 = np.pad(des1, (0, diff), 'constant')
        # Compute Euclidean distance
        return self.computeEucledian(des1, des2)
