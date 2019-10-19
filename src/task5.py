from sklearn import preprocessing, svm

from src.common.latentSemanticsHelper import getSemanticsFromFolder, getParams
from src.dimReduction.dimRedHelper import getQueryImageRep
from src.common.latentSemanticsHelper import getParams, getSemanticsFromFolder
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from src.common.helper import getImagePathsWithLabel
from src.dimReduction.dimRedHelper import getQueryImageRepList, getQueryImageRep
from src.models.enums.models import ModelType
from src.dimReduction.dimRedHelper import getDataMatrix
from src.dimReduction.SVD import SVD
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import sys
def initTask5_2(folderPath,  imagePath):
    classificatonMeta = {
        "dorsal" : "palmar",
        "left" : "right",
        "accessories": "without-accessories",
        "male": "female",
        "palmar": "dorsal",
        "right": "left",
        "without-accessories": "accessories",
        "female": "male"
    }

    _, modelType, dimRedType, k, label = getParams(folderPath)
    # print(folderPath)
    u, vt = getSemanticsFromFolder(folderPath)
    u = preprocessing.scale(u)
    uMean = np.mean(u, axis=0)
    maxdis, mindis = -10000000, 10000000
    for item in u:
        d = np.linalg.norm(item-uMean)
        mindis = min(mindis, d)
        maxdis = max(maxdis, d)
    print(mindis, maxdis, label)

    queryImage = getQueryImageRep(vt, imagePath, modelType)
    queryImageNormalised = preprocessing.scale(queryImage)
    qdis = np.linalg.norm(queryImageNormalised - uMean)
    print(qdis)


def initTask5(folderPath, imagePath):
    classificatonMeta = {
        "dorsal" : "palmar",
        "left" : "right",
        "accessories": "without-accessories",
        "male": "female",
        "palmar": "dorsal",
        "right": "left",
        "without-accessories": "accessories",
        "female": "male"
    }

    _, modelType, dimRedType, k, label = getParams(folderPath)
    u, vt = getSemanticsFromFolder(folderPath)
    uNomalised = preprocessing.scale(u)

    oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.1)
    oc_svm_clf.fit(uNomalised)

    queryImage = getQueryImageRep(vt, imagePath, modelType)
    queryImageNormalised = preprocessing.scale(queryImage)
    queryPrediction = oc_svm_clf.predict(queryImageNormalised)

    if len(queryPrediction) < 1:
        raise ValueError("Query prediction not available")

    if queryPrediction[0] == 1:
        print("Predicted the image as '{}'".format(label))
    else:
        print("Predicted the image as '{}'".format(classificatonMeta[label]))
