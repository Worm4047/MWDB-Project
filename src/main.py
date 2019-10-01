import os
import cv2
import glob
import csv
import numpy as np
import datetime
from models import ColorMoments
from models import SIFT
from models import LBP
# from models import ColorMoments
from comparators import CMComparator
from computeColorMoments import computeColorMoments
from siftFeatureExtractionUtil import computeSIFTFeatures, getSIFTFeatures, getSIFTDistance
from common.helper import plotFigures, getImageName
import matplotlib.pyplot as plt

def getColorMomentsFeature(colorMomentsFeatures, imagePath):
    for colorMommentsFeature in colorMomentsFeatures:
        if colorMommentsFeature[0] == imagePath:
            return colorMommentsFeature

    raise ValueError("Image not found in database")

#### Function or user interaction ####

def getModelAndTask():
   
    while True:
        print("-----------------------------------------------------------------------------------------")
        print("-> Please the task you would like to perform")
        print("-> 1. Task 1")
        print("-> 2. Task 2")
        print("-> 3. Task 3")
        print("-> 4. Task 4")
        print("-> 5. Task 5")
        taskType = input("-> Please enter the number: ")

        if taskType in ['1','2', '3', '4', '5']: break
    while True:
        print("-----------------------------------------------------------------------------------------")
        print("-> Please the model to use")
        print("-> 1. Color moments")
        print("-> 2. SIFT")
        print("-> 3. LBP")
        print("-> 4. HOG")
        modelType = input("-> Please enter the number: ")

        if modelType == "1" or modelType == "2" or modelType == "3" or modelType == "4": break


    return modelType, taskType

def getDatabasePath():
    while True:
        print("-----------------------------------------------------------------------------------------")
        dbPath = input("-> Please enter the database path: ")

        if len(dbPath) > 4: break

    return dbPath

def getKFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        k = input("-> Please enter the number top latent semantics (k) ")
        try:
            val = int(k)
            if(val > 0): return val
            else: continue
        except ValueError:
            continue

def getMFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        m = input("-> Please enter the number of similar images (m) ")
        try:
            val = int(m)
            if(val > 0): return val
            else: continue
        except ValueError:
            continue

def getDimTechniqueFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        print("-> Please enter the Dimensionality reduction technique to use. ")
        print("-> 1. PCA")
        print("-> 2. SVD")
        print("-> 3. NMF")
        print("-> 4. LDA")
        dimType = input("-> Please enter the number: ")

        if dimType == "1" or dimType == "2" or dimType == "3" or dimType == "4": return dimType

def getLabelFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        label = input("-> Please enter the label for the image")
        print("-> 1. Left-Handed")
        print("-> 2. Right-Handed")
        print("-> 3. Dorsal")
        print("-> 4. Palmer")
        print("-> 5. With-Accessories")
        print("-> 6. Without-Accessories")
        print("-> 7. Male")
        print("-> 8. Female")
        label = input("-> Please enter the number: ")

        if label in ['1','2','3','4','5','6','7','8'] :return label   
        else: print(' Incorrect value ') 

def getImagePathFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        imagePath = input("-> Please enter the full path of the image: ")

        if os.path.exists(imagePath): return imagePath
        else:
            print("Invalid path")

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

def getLatentSemantics(modelType, dimTechnique, k):
    pass

def getSimilarImages(modelType, dimTechnique, k, m):
    pass

def getLatentSemanticsForLabel(modelType, dimTechnique, k, label):
    pass

def getSimilarImagesForLabel(modelType, dimTechnique, k, m, label):
    pass

def init():

    modelType, taskType = getModelAndTask()
    k = getKFromUser()
    dimTechnique = getDimTechniqueFromUser()

    if taskType in ['1','2']:

        if taskType == '1':
            getLatentSemantics(modelType, dimTechnique, k)

        if taskType == '2':
            imagepath = getImagePathFromUser()
            m = getMFromUser()
            getSimilarImages(modelType, dimTechnique, k, m)
    
    elif taskType in ['3','4']:

        label = getLabelFromUser()

        if taskType == '3':
            getLatentSemanticsForLabel(modelType, dimTechnique, k, label)
        
        if taskType == '4':
            imagepath = getImagePathFromUser()
            m = getMFromUser()
            getSimilarImagesForLabel(modelType, dimTechnique, k, m, label)            
            
if __name__ == "__main__":

    init()
