from src.dimReduction.dimRedHelper import getDataMatrix
from src.models.enums.models import ModelType
from src.dimReduction.SVD import SVD
from src.dimReduction.LDA import LDA
import numpy as np

def init():
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.CM).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.HOG).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.LBP).shape)
    dm = getDataMatrix("/home/worm/Desktop/ASU/CSE 515/Phase#2/MWDB/src/images", ModelType.CM)

    LDA(dm).getDecomposition()

if __name__ == "__main__":
    init()