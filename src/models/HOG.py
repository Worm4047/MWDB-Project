from src.models.interfaces.Model import Model
from skimage.feature import hog

class HOG(Model):
    def __init__(self, grayScale, numBins, CellSize, BlockSize):
        self.numBins = numBins
        self.CellSize = CellSize
        self.BlockSize = BlockSize
        self.computeFeatureVector(grayScale)

    def computeFeatureVector(self, grayScaleImage):
        self.featureVector = hog(grayScaleImage, orientations=self.numBins,
                             pixels_per_cell=(self.CellSize, self.CellSize),
                             cells_per_block=(self.BlockSize, self.BlockSize),
                             block_norm='L2-Hys',feature_vector=True, multichannel=True)

    def getFeatures(self):
        return self.featureVector

    def compare(self, hogModel):
        if not isinstance(hogModel, Model):
            raise ValueError("Not a HOG model")

        return 0
