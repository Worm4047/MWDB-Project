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
    csvFilePath = "D:/studies/multimedia and web databases/project/HandInfo.csv"
    databasePath = "D:/studies/multimedia and web databases/project/CSE 515 Fall19 - Smaller Dataset\CSE 515 Fall19 - Smaller Dataset/"
    dorsalImageIds = getImagePathsWithLabel("dorsal", csvFilePath, databasePath)
    print(len(dorsalImageIds))

    dmSIFT = getDataMatrix(dorsalImageIds, ModelType.HOG, "dorsal")
    u, vt = SVD(dmSIFT, 10).getDecomposition()

    oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.1)
    oc_svm_clf.fit(u)
    oc_svm_preds = oc_svm_clf.predict(u)

    palmarImagePaths = getImagePathsWithLabel("palmar", csvFilePath, databasePath)
    palmarImagePaths = palmarImagePaths[0:500]
    palmarKspace = getQueryImageRepList(vt, palmarImagePaths, ModelType.HOG)
    oc_svm_predsT = oc_svm_clf.predict(palmarKspace)

if __name__ == "__main__":
    init()