from sklearn import preprocessing, svm

from src.common.latentSemanticsHelper import getSemanticsFromFolder, getParams
from src.dimReduction.dimRedHelper import getQueryImageRep


def initTask5(folderPath, csvFilePath, imagePath):
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
