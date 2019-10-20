from src.dimReduction.dimRedHelper import getDataMatrix
from src.models.enums.models import ModelType
from src.dimReduction.SVD import SVD
import numpy as np
from src.dimReduction.dimRedHelper import getQueryImageRep
from src.common.latentSemanticsHelper import saveSemantics, getParams, getSemanticsFromFolder
from src.dimReduction.enums.reduction import ReductionType
from src.common.helper import getImagePathsWithLabel

def init():
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.CM).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.HOG).shape)
    # print(getDataMatrix("/Users/yvtheja/Documents/TestHands", ModelType.LBP).shape)

    # print(getDataMatrix(None, ModelType.SIFT, directoryPath="/Users/yvtheja/Documents/TestHands").shape)
    # # Dummy with k = 13
    # print(getQueryImageRep(np.ones((13, 1290)), "/Users/yvtheja/Documents/TestHands/Hand_0000002.jpg", ModelType.SIFT).shape)
    # # Dummy with k = 17
    # print(getQueryImageRep(np.ones((17, 1728)), "/Users/yvtheja/Documents/TestHands/Hand_0000002.jpg", ModelType.CM).shape)

    # saveSemantics("testfolder", ModelType.SIFT, "dorsal", ReductionType.SVD, 10, np.ones((23, 10)), np.zeros((10, 1200)))
    # u, v = getSemanticsFromFolder("/Users/yvtheja/Documents/ASU/MWDB/src/store/testfolder_SIFT_SVD_10_dorsal")
    # print(u.shape)
    # print(v.shape)
    # params = getParams("/Users/yvtheja/Documents/ASU/MWDB/src/store/testfolder_SIFT_SVD_10_dorsal")
    # print(params)

    csvFilePath = "/Users/yvtheja/Documents/HandInfo.csv"
    databasePath = "/Users/yvtheja/Documents/Hands"
    # u, vt = getSemanticsFromFolder(folderPath)
    dorsalImageIds = getImagePathsWithLabel(3, csvFilePath, databasePath)
    dorsalImageIds = dorsalImageIds[0: 23]
    dmSIFT = getDataMatrix(dorsalImageIds, ModelType.CM, 3)
    SVD(dmSIFT, 10).getDecomposition()

    # Refer this for calling DimRed
    # SVD(np.array([])).getDecomposition()

if __name__ == "__main__":
    init()