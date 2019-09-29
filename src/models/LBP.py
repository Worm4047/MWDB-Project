from src.models.interfaces.Model import Model
from skimage.feature import local_binary_pattern
import numpy as np
import math

class LBP(Model):
    def __init__(self, grayScaleImage, blockSize, numPoints, radius, eps=1e-7):
        self.validateImage(grayScaleImage)
        super(LBP, self).__init__()
        self.getLBPFeature(grayScaleImage, numPoints, radius, eps)

    def getLBPFeature(self, grayScaleImage, numPoints, radius, eps):
        lbpFeatures = local_binary_pattern(grayScaleImage, numPoints, radius, method="uniform");
        (hist, _) = np.histogram(lbpFeatures.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
        hist = hist.astype(np.float)
        hist /= (hist.sum() + eps)
        self.featureVector = hist

    def validateImage(self, image):
        if image.shape is None:
            raise ValueError("Not a np array")
        if len(image.shape) != 2:
            raise ValueError("Invalid Image")

    def getFeatures(self):
        return self.featureVector

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
