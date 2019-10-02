import glob
import numpy as np
from models import SIFT
# from models import ColorMoments
from comparators import CMComparator
from src.models.enums.models import ModelType
from src.dimReduction.dimRedHelper import getDataMatrix
from src.common.helper import getImagePathFromUser
from src.common.helper import getModelFromUser

#### Function or user interaction ####


# def getDimReductionMethodFromUser():
#     while True:
#         print("-----------------------------------------------------------------------------------------")
#         print("-> Please the model to use")
#         print("-> 1. SVD")
#         print("-> 2. LBP")
#         print("-> 3. HOG")
#         print("-> 4. SIFT")
#         modelType = input("-> Please enter the number: ")
#
#         if modelType == "1" or modelType == "2" or modelType == "3" or modelType == "4": break
#
#     return ModelType(int(modelType))


#### End of user interaction functons
# def getDistancesWithCM(imagePath):
#     distances = []
#     with open('featureVectorsStore/colorMomentsFeatures.csv', newline='') as csvfile:
#         colorMomentsFeatures = csv.reader(csvfile, delimiter=',')
#         dbImg = cv2.imread(imagePath, cv2.IMREAD_COLOR)
#         dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
#         dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)
#         queryImageFeatureVector = dbImageColorMomments.featureVector

#         for index, colorMommentsFeature in enumerate(colorMomentsFeatures):
#             print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
#             dbImagePath, imageFeatureVectorString, _ = colorMommentsFeature
#             imageFeatureVectorFlat = np.array(imageFeatureVectorString.split(',')).astype(np.float)
#             imageFeatureVector = imageFeatureVectorFlat.reshape((12, 16, 9))
#             momentsY = np.concatenate((imageFeatureVector[:, :, 0], imageFeatureVector[:, :, 3], imageFeatureVector[:, :, 6]))

#             distance = np.linalg.norm(queryImageFeatureVector - imageFeatureVector)

#             distances.append((dbImagePath, distance))

#     return distances

# def getDistancesWithSIFT(imagePath):
#     distances = []
#     queryImageKp, queryImageDes = getSIFTFeatures(imagePath)
#     imageDesFilePaths = glob.glob("featureVectorsStore/siftStore/*_des.npy")

#     for index, imageDesFilePath in enumerate(imageDesFilePaths):
#         print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
#         imageDes = np.load(imageDesFilePath)
#         imageName = "_".join(os.path.basename(imageDesFilePath).split('.')[0].split('_')[:-1])
#         imageFileName = imageName + ".jpg"

#         distances.append((imageFileName, getSIFTDistance(imageDes, queryImageDes)))

#     return distances

# def getSimilarImages(databasePath, imagePath, modelType = 1, k=15):
#     distances = []
#     if modelType == "1": distances = getDistancesWithCM(imagePath)
#     elif modelType == "2": distances = getDistancesWithSIFT(imagePath)

#     distances.sort(key=lambda x: x[1])
#     similarImageDistances = distances[0: k]
#     similarImagesMap = {}
#     for index, similarImageDistance in enumerate(similarImageDistances):
#         imageFileName = similarImageDistance[0]
#         imagePath = os.path.join(databasePath, imageFileName)
#         similarImagesMap["Distance: {}".format(round(similarImageDistance[1], 2))] = cv2.cvtColor(cv2.imread(imagePath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

#     plotFigures(similarImagesMap, 5)

# def showColorMomentsFeatureVector(imagePath):
#     dbImg = cv2.imread(imagePath, cv2.IMREAD_COLOR)
#     dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
#     dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)
#     print("Shape of the feature vector: {}".format(dbImageColorMomments.featureVector.shape))
#     print("Value of the flattened feature vector: [{}]".format(",".join(dbImageColorMomments.featureVector.flatten().astype(np.str))))

# def showSIFTFeatureVector(imagePath):
#     dbImg = cv2.imread(imagePath, cv2.IMREAD_COLOR)
#     imageGray = cv2.cvtColor(dbImg, cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d.SIFT_create()
#     keyPoints, descriptors = sift.detectAndCompute(imageGray, None)
#     print("Number of features extracted: {}".format(len(keyPoints)))
#     for index, keyPoint in enumerate(keyPoints):
#         print("X: {}, Y: {}, Scale: {}, Orientation: {}, HOG: [{}]".format(
#             keyPoint.pt[1], keyPoint.pt[0], keyPoint.size, keyPoint.angle, ",".join(descriptors[index].flatten().astype(np.str))))

# def showFeatureVector(imagePath, modeltype):
#     if(modeltype == "1"): showColorMomentsFeatureVector(imagePath)
#     if(modeltype == "2"): showSIFTFeatureVector(imagePath)

# def extractAndStoreFeatures(databasePath, modelType):
#     if modelType == "1": computeColorMoments(databasePath)
#     else: computeSIFTFeatures(databasePath);

def init():
    # k = getKFromUser()
    # dimTechnique = getDimTechniqueFromUser()
    # imagePath = "/Users/yvtheja/Documents/TestHands"
    imagePath = getImagePathFromUser()
    modelType = getModelFromUser()
    matrix = getDataMatrix(imagePath, modelType)
    print(matrix.shape)

    # if taskType in ['1','2']:
    #
    #     if taskType == '1':
    #         getLatentSemantics(modelType, dimTechnique, k)
    #
    #     if taskType == '2':
    #         imagepath = getImagePathFromUser()
    #         m = getMFromUser()
    #         getSimilarImages(modelType, dimTechnique, k, m)
    #
    # elif taskType in ['3','4']:
    #
    #     label = getLabelFromUser()
    #
    #     if taskType == '3':
    #         getLatentSemanticsForLabel(modelType, dimTechnique, k, label)
    #
    #     if taskType == '4':
    #         imagepath = getImagePathFromUser()
    #         m = getMFromUser()
    #         getSimilarImagesForLabel(modelType, dimTechnique, k, m, label)


if __name__ == "__main__":
    init()
