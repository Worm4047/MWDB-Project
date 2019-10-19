import numpy as np
import os
from src.models.enums.models import ModelType
from src.dimReduction.enums.reduction import ReductionType

def saveSemantics(imageDirName, modelType, label, dimRedTech, k, U, V, dirPath="store/latentSemantics"):
    if not isinstance(modelType, ModelType):
        raise ValueError("Invalid model type")

    if not isinstance(dimRedTech, ReductionType):
        raise ValueError("Invalid Reduction type")

    if not isinstance(U, np.ndarray) or not isinstance(V, np.ndarray):
        raise ValueError("Invalid arguments U and V")

    # "{ImageDirName}_{modelType}_{dimRedTechnique}_{K}_{label}.csv"
    folderName = "{}_{}_{}_{}_{}".format(imageDirName, modelType.name, dimRedTech.name, k, label)
    folderPath = os.path.join(dirPath, folderName)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    uFilePath = os.path.join(folderPath, "U.csv")
    vFilePath = os.path.join(folderPath, "V.csv")
    np.savetxt(uFilePath, U, delimiter=",")
    np.savetxt(vFilePath, V, delimiter=",")

def getSemanticsFromFolder(folderPath):
    if not os.path.isdir(folderPath):
        return None

    uFilePath = os.path.join(folderPath, "U.csv")
    vFilePath = os.path.join(folderPath, "V.csv")
    if not os.path.exists(uFilePath) or not os.path.exists(vFilePath):
        return None
    print(uFilePath, vFilePath)
    return np.genfromtxt(uFilePath, delimiter=','), np.genfromtxt(vFilePath, delimiter=',')

def getParams(folderPath):
    # "{ImageDirName}_{modelType}_{dimRedTechnique}_{K}_{label}.csv"
    folderName = os.path.basename(folderPath)

    paramsArray = folderName.split('_')
    if len(paramsArray) != 5:
        raise ValueError("Invalid latent semantic file")

    return paramsArray[0], ModelType[paramsArray[1]], ReductionType[paramsArray[2]], int(paramsArray[3]), paramsArray[4]
