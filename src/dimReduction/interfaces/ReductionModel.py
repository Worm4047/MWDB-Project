import numpy as np

class ReductionModel:
    def __init__(self, dataMatrix):
        if not isinstance(dataMatrix, np.ndarray):
            raise ValueError("Data matrix should be a numpy array")

        self.dataMatrix = dataMatrix

    def getDecomposition(self):
        pass