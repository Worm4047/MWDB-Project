import sys

import numpy

from src.common.helper import getImagePathsWithLabel

from src.dimReduction.LDA import LDA
from src.archive.dimRedHelper import getDataMatrix
from src.models.enums.models import ModelType


def init():
    csvFilePath = "/Users/studentworker/PycharmProjects/mwdb/Phase_1/test/HandInfo.csv"
    databasePath = "/Users/studentworker/PycharmProjects/mwdb/Phase_1/test/Hands_original"
    dorsalImageIds = getImagePathsWithLabel(1, csvFilePath, databasePath)
    dorsalImageIds = dorsalImageIds[0: 100]
    print(dorsalImageIds)

    dm = getDataMatrix(dorsalImageIds, ModelType.HOG, "", None)
    # u, s, vt = LDA(dm, 0).getDecomposition()
    # u_np = numpy.array(u)
    # v_np = numpy.array(vt)
    numpy.set_printoptions(threshold=sys.maxsize)
    print(dm.shape)
    print(dm)
    # print("ABC")
    # print(v_np.shape)
    # u_reduced = numpy.matmul(u_np, v_np)
    # utransformed = numpy.matmul(u_np, v_np)
    # utransformed = scale(u_np)
    # print(utransformed.shape)
    # oc_svm_clf = svm.OneClassSVM(gamma=0.2, kernel='rbf', nu=0.1)
    # oc_svm_clf.fit(utransformed)
    # oc_svm_preds = oc_svm_clf.predict(u)
    U, V = LDA(dm, 20).getDecomposition()
    # query_image_features = getQueryImageRep(V, "/Users/studentworker/PycharmProjects/mwdb/Phase_1/test/Hands/Hand_0000002.jpg", ModelType.HOG)
    # list = comparisonHelper.getMSimilarImages(U, query_image_features, 3, ModelType.HOG)
    print(numpy.array(U).shape)
    print(numpy.array(V).shape)
    # plotFigures(list)

    #
    # palmarKspace = getQueryImageRep(vt, "/Users/studentworker/PycharmProjects/mwdb/Phase 1/test/Hands/Hand_0000002.jpg", ModelType.HOG)
    # palmarKspaceTranspose = scale(palmarKspace)
    # palmarKspaceTranspose = palmarKspaceTranspose.reshape(-1, 1)
    # print(palmarKspaceTranspose.shape)
    # oc_svm_predsT = oc_svm_clf.predict(palmarKspaceTranspose.T)
    #
    # print(oc_svm_predsT)
    print("Boom")

if __name__ == "__main__":
    init()



 # csvFilePath = "/Users/studentworker/PycharmProjects/mwdb/Phase 1/test/HandInfo.csv"
 #    databasePath = "/Users/studentworker/PycharmProjects/mwdb/Phase 1/test/Hands_original"
 #    dorsalImageIds = getImagePathsWithLabel("palmar", csvFilePath, databasePath)
 #    dorsalImageIds = dorsalImageIds[0: 100]
 #
 #    print(dorsalImageIds)
 #
 #    dm = getDataMatrix(dorsalImageIds, ModelType.HOG)
 #    # u, v = LDA(dm, 10).getDecomposition()
 #    # u_np = numpy.array(u)
 #    # v_np = numpy.array(v)
 #    # numpy.set_printoptions(threshold=sys.maxsize)
 #    # print(u_np.shape)
 #    # print("ABC")
 #    # print(v_np.shape)
 #    # utransformed = scale(u.T)
 #    oc_svm_clf = svm.OneClassSVM(gamma=0.2, kernel='rbf', nu=0.1)
 #    oc_svm_clf.fit(dm)
 #    # oc_svm_preds = oc_svm_clf.predict(u)
 #
 #    # palmarKspace = getQueryImageRep(v, "/Users/studentworker/PycharmProjects/mwdb/Phase 1/test/Hands/Hand_0000002.jpg", ModelType.HOG)
 #    query = getImageFeatures("/Users/studentworker/PycharmProjects/mwdb/Phase 1/test/Hands/Hand_0000002.jpg", ModelType.HOG)
 #    # palmarKspaceTranspose = scale(palmarKspace)
 #    palmarKspaceTranspose = query.reshape(-1, 1)
 #    print(palmarKspaceTranspose.shape)
 #    oc_svm_predsT = oc_svm_clf.predict(palmarKspaceTranspose.T)
 #
 #    print(oc_svm_predsT)
 #    print("Boom")