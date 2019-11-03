from src.common.latentSemanticsSaver import LatentSemanticSaver
from src.dimReduction.enums.reduction import ReductionType
from src.dimReduction.SVD import SVD
from src.dimReduction.PCA import PCA
from src.dimReduction.NMF import NMF
from src.dimReduction.LDA import LDA

class LatentSemantic:
    def getLatentSemantic(k, decompType, dataMatrix, modelType, label, imageDirName, imagePaths):
        folderName = "{}_{}_{}_{}_{}".format(imageDirName, modelType.name, decompType.name, k, label)
        lsPath = LatentSemanticSaver().getLatentSemanticPath(modelType, decompType, k, label)
        latent_semantic = LatentSemanticSaver().getSemanticsFromFolder(lsPath)
        if latent_semantic is None:
            if decompType == ReductionType.SVD:
                u, v = SVD(dataMatrix, k).getDecomposition()
                latent_semantic = u, v
            elif decompType == ReductionType.PCA:
                latent_semantic = PCA(dataMatrix, k).getDecomposition()
            elif decompType == ReductionType.NMF:
                latent_semantic = NMF(dataMatrix, k).getDecomposition()
            elif decompType == ReductionType.LDA:
                # latent_semantic = LDA(dataMatrix, k).getDecomposition()
                raise ValueError("LDA is not supported yet for phase 3")
            else:
                print("Check later")
                return None
            print("Image path example ", imagePaths[0])
            LatentSemanticSaver().saveSemantics(modelType, label, decompType, k,
                                                latent_semantic[0], latent_semantic[1], imagePaths=imagePaths)
        return latent_semantic
