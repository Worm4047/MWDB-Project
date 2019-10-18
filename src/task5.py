from src.common.helper import getImagePathsWithLabel
from src.dimReduction.dimRedHelper import getQueryImageRepList
from src.models.enums.models import ModelType
from src.dimReduction.dimRedHelper import getDataMatrix
from src.dimReduction.SVD import SVD
import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing

from src.common.helper import getImagePathsWithLabel
from src.dimReduction.SVD import SVD
from src.dimReduction.dimRedHelper import getDataMatrix
from src.dimReduction.dimRedHelper import getQueryImageRepList
from src.models.enums.models import ModelType


def initTask5(folderPath, csvFilePath, imagePath):
    # _, modelType, dimRedType, k, label = getParams(folderPath)
    csvFilePath = "/Users/yvtheja/Documents/HandInfo.csv"
    databasePath = "/Users/yvtheja/Documents/Hands"
    # u, vt = getSemanticsFromFolder(folderPath)
    dorsalImageIds = getImagePathsWithLabel("dorsal", csvFilePath, databasePath)
    dorsalImageIds = dorsalImageIds[0: 23]

    dmSIFT = getDataMatrix(dorsalImageIds, ModelType.CM, "dorsal")
    u, s, vt = SVD(dmSIFT, 10).getDecomposition()
    u = preprocessing.scale(u)
    ucentroid = np.mean(u, axis=0)
    distances = []
    for row in u:
        distances.append(np.linalg.norm(row - ucentroid))

    print("U Dorsal | mean: {} | std: {}".format(np.mean(distances), np.std(distances)))


    palmarImagePaths = getImagePathsWithLabel("palmar", csvFilePath, databasePath)
    palmarImagePaths = palmarImagePaths[0:22]
    palmarKspace = getQueryImageRepList(vt, palmarImagePaths, ModelType.CM)
    palmarKspace = preprocessing.scale(palmarKspace)
    distances = []
    for row in palmarKspace:
        distances.append(np.linalg.norm(row - ucentroid))

    print("Q palmar | mean: {} | std: {}".format(np.mean(distances), np.std(distances)))

    palmarImagePaths = getImagePathsWithLabel("dorsal", csvFilePath, databasePath)
    palmarImagePaths = palmarImagePaths[0:22]
    palmarKspace = getQueryImageRepList(vt, palmarImagePaths, ModelType.CM)
    palmarKspace = preprocessing.scale(palmarKspace)
    distances = []
    for row in palmarKspace:
        distances.append(np.linalg.norm(row - ucentroid))

    print("Q dorsal | mean: {} | std: {}".format(np.mean(distances), np.std(distances)))
    print("Boom")

if __name__ == "__main__":
    initTask5(None, None, None)