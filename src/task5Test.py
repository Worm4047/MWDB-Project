import pandas as pd
from src.common.helper import getImagePathsWithLabel
from src.common.imageHelper import getGrayScaleImage
from src.dimReduction.dimRedHelper import getDataMatrix
from src.models.enums.models import ModelType
from src.dimReduction.SVD import SVD
from src.dimReduction.dimRedHelper import getQueryImageRepList
import os
from sklearn import svm
import numpy as np

def init():
    csvFilePath = "/Users/yvtheja/Documents/HandInfo.csv"
    databasePath = "/Users/yvtheja/Documents/TestHands"
    dorsalImageIds = getImagePathsWithLabel("dorsal", csvFilePath, databasePath)
    dorsalImageIds = dorsalImageIds[0: 50]

    dmSIFT = getDataMatrix(dorsalImageIds, ModelType.SIFT, "dorsal")
    u, s, vt = SVD(dmSIFT, 10).getDecomposition()

    oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.1)
    oc_svm_clf.fit(u)
    oc_svm_preds = oc_svm_clf.predict(u)

    palmarImagePaths = getImagePathsWithLabel("palmar", csvFilePath, databasePath)
    palmarImagePaths = palmarImagePaths[0:500]
    palmarKspace = getQueryImageRepList(vt, palmarImagePaths, ModelType.SIFT)
    oc_svm_predsT = oc_svm_clf.predict(palmarKspace)

    print("Boom")

if __name__ == "__main__":
    init()