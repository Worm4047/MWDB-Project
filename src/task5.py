from src.common.latentSemanticsHelper import getParams, getSemanticsFromFolder
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from src.common.helper import getImagePathsWithLabel
from src.dimReduction.dimRedHelper import getQueryImageRepList
from src.models.enums.models import ModelType
from src.dimReduction.dimRedHelper import getDataMatrix
from src.dimReduction.SVD import SVD

def initTask5(folderPath, csvFilePath, imagePath):
    # _, modelType, dimRedType, k, label = getParams(folderPath)
    csvFilePath = "/Users/yvtheja/Documents/HandInfo.csv"
    databasePath = "/Users/yvtheja/Documents/Hands"
    # u, vt = getSemanticsFromFolder(folderPath)
    dorsalImageIds = getImagePathsWithLabel("dorsal", csvFilePath, databasePath)
    dorsalImageIds = dorsalImageIds[0: 22]

    dmSIFT = getDataMatrix(dorsalImageIds, ModelType.SIFT)
    u, s, vt = SVD(dmSIFT, 10).getDecomposition()

    ss = StandardScaler().fit(u)
    uTransformed = ss.transform(u)

    oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.1)
    oc_svm_clf.fit(uTransformed)
    oc_svm_preds = oc_svm_clf.predict(uTransformed)


    palmarImagePaths = getImagePathsWithLabel("palmar", csvFilePath, databasePath)
    palmarImagePaths = palmarImagePaths[0:26]
    palmarKspace = getQueryImageRepList(vt, palmarImagePaths, ModelType.SIFT)
    oc_svm_predsT = oc_svm_clf.predict(palmarKspace)

if __name__ == "__main__":
    initTask5(None, None, None)