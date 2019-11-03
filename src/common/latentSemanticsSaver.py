import numpy as np
import os
from src.models.enums.models import ModelType
from src.dimReduction.enums.reduction import ReductionType
from src.common.enums.labels import LabelType

class LatentSemanticSaver:
    def __init__(self):
        self.dirPath = "store/latentSemantics"

    def saveSemantics(self, modelType, label, dimRedTech, k, U, V, imagePaths):
        if not isinstance(modelType, ModelType):
            raise ValueError("Invalid model type")

        if not isinstance(dimRedTech, ReductionType):
            raise ValueError("Invalid Reduction type")

        if not isinstance(U, np.ndarray) or not isinstance(V, np.ndarray):
            raise ValueError("Invalid arguments U and V")

        if not isinstance(label, LabelType):
            raise ValueError("Invalid label type. It should be of type LabelType.")

        # "{ImageDirName}_{modelType}_{dimRedTechnique}_{K}_{label}.csv"
        folderName = "{}_{}_{}_{}".format(modelType.name, dimRedTech.name, k, label.name)
        folderPath = os.path.join(self.dirPath, folderName)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        uFilePath = os.path.join(folderPath, "U.csv")
        vFilePath = os.path.join(folderPath, "V.csv")
        imagePathsFilePath = os.path.join(folderPath, "imagenames.csv")
        np.savetxt(uFilePath, U, delimiter=",")
        np.savetxt(vFilePath, V, delimiter=",")
        np.savetxt(imagePathsFilePath, imagePaths, fmt="%s")

    def getLatentSemanticPath(self, modelType, dimRedTech, k, label):
        if not isinstance(label, LabelType):
            raise ValueError("Invalid label type. It should be of type LabelType.")
        
        return "store/latentSemantics/{}_{}_{}_{}".format(modelType.name, dimRedTech.name, k, label.name)

    def getSemanticsFromFolder(self, folderPath):
        if not os.path.isdir(folderPath):
            return None

        uFilePath = os.path.join(folderPath, "U.csv")
        vFilePath = os.path.join(folderPath, "V.csv")
        imagePathsFilePath = os.path.join(folderPath, "imagenames.csv")
        if not os.path.exists(uFilePath) or not os.path.exists(vFilePath):
            return None
        return np.genfromtxt(uFilePath, delimiter=','), np.genfromtxt(vFilePath, delimiter=','), np.genfromtxt(imagePathsFilePath,dtype=None, delimiter="\n")

    def getParams(self, folderPath):
        # "{modelType}_{dimRedTechnique}_{K}_{label}.csv"
        folderName = os.path.basename(folderPath)

        paramsArray = folderName.split('_')
        if len(paramsArray) != 4:
            raise ValueError("Invalid latent semantic file")

        return ModelType[paramsArray[0]], ReductionType[paramsArray[1]], int(paramsArray[2]), paramsArray[3]
