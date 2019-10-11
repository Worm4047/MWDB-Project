from src.dimReduction.dimRedHelper import getDataMatrix
from src.models.enums.models import ModelType
from src.dimReduction.SVD import SVD
import numpy as np
from src.dimReduction.dimRedHelper import getQueryImageRep

def init():
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.CM).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.HOG).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.LBP).shape)

    print(getDataMatrix(None, ModelType.SIFT, directoryPath="/Users/yvtheja/Documents/TestHands").shape)
    # Dummy with k = 13
    print(getQueryImageRep(np.ones((13, 1290)), "/Users/yvtheja/Documents/TestHands/Hand_0000002.jpg", ModelType.SIFT).shape)
    # Dummy with k = 17
    print(getQueryImageRep(np.ones((17, 1728)), "/Users/yvtheja/Documents/TestHands/Hand_0000002.jpg", ModelType.CM).shape)


    # Refer this for calling DimRed
    # SVD(np.array([])).getDecomposition()

if __name__ == "__main__":
    init()