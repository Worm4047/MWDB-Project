from src.common.latentSemanticsHelper import getParams, getSemanticsFromFolder
from sklearn import svm

def initTask5(folderPath, imagePath):
    _, modelType, dimRedType, k, label = getParams(folderPath)
    u, v = getSemanticsFromFolder(folderPath)

    oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.1)
    oc_svm_clf.fit(u)
    oc_svm_preds = oc_svm_clf.predict(u)

    palmarImagePaths = getImagePathsWithLabel("palmar", csvFilePath, databasePath)
    palmarImagePaths = palmarImagePaths[0:26]
    palmarKspace = getQueryImageRepList(vt, palmarImagePaths, ModelType.SIFT)
    oc_svm_predsT = oc_svm_clf.predict(palmarKspace)

