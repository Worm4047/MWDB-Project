from src.dimReduction.dimRedHelper import getDataMatrix
from src.models.enums.models import ModelType
from src.dimReduction.SVD import SVD
import numpy as np

def init():
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.CM).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.HOG).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.LBP).shape)

    print(getDataMatrix(None, ModelType.SIFT, directoryPath="/Users/yvtheja/Documents/TestHands").shape)

    # Refer this for calling DimRed
    # SVD(np.array([])).getDecomposition()

if __name__ == "__main__":
    init()