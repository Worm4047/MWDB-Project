from src.models.enums.models import ModelType
from src.dimReduction.enums.reduction import ReductionType
import os
import cv2
import pandas as pd
import numpy as np


def getTaskFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        print("-> Please the task you would like to perform")
        print("-> 1. Task 1")
        print("-> 2. Task 2")
        print("-> 3. Task 3")
        print("-> 4. Task 4")
        print("-> 5. Task 5")
        print("-> 6. Task 6")
        print("-> 7. Task 7")
        print("-> 8. Task 8")
        taskType = input("-> Please enter the number: ")

        if taskType in ['1','2', '3', '4', '5', '6', '7', '8']: break

    return taskType


def getModelFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        print("-> 1. Color moments")
        print("-> 2. LBP")
        print("-> 3. HOG")
        print("-> 4. SIFT")
        modelType = input("-> Please enter the number: ")

        if modelType == "1" or modelType == "2" or modelType == "3" or modelType == "4": break

    return ModelType(int(modelType))


def getDimTechniqueFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        print("-> Please enter the Dimensionality reduction technique to use. ")
        print("-> 1. SVD")
        print("-> 2. PCA")
        print("-> 3. LDA")
        print("-> 4. NMF")
        dimType = input("-> Please enter the number: ")

        if dimType == "1" or dimType == "2" or dimType == "3" or dimType == "4": break

    return ReductionType(int(dimType))


def getDatabasePath():
    while True:
        print("-----------------------------------------------------------------------------------------")
        dbPath = input("-> Please enter the image folder path: ")

        if len(dbPath) > 4: break

    return dbPath

def getDatabasePathFor11k():
    while True:
        print("-----------------------------------------------------------------------------------------")
        dbPath = input("-> Please enter the image folder path for 11K images: ")

        if len(dbPath) > 4: break

    return dbPath

def getMetadataPath():
    while True:
        print("-----------------------------------------------------------------------------------------")
        dbPath = input("-> Please enter the Metadata folder path: ")

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
        print("-----------------------------------------------------------------------------------------")
        m = input("-> Please enter the number of similar images (m) ")
        val = int(m)
        if(val > 0): return val


def getLabelFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")

        print("-> 1. Left-Handed")
        print("-> 2. Right-Handed")
        print("-> 3. Dorsal")
        print("-> 4. Palmer")
        print("-> 5. With-Accessories")
        print("-> 6. Without-Accessories")
        print("-> 7. Male")
        print("-> 8. Female")
        label = input("-> Please enter the label for the image")

        if label in ['1','2','3','4','5','6','7','8']: return int(label)
        else: print(' Incorrect value ')


def getImagePathFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        imagePath = input("-> Please enter the full path of the image: ")

        if os.path.exists(imagePath): return imagePath
        else:
            print("Invalid path")

def getSubjectId():
    while True:
        print("----------------------------------------------------------------------------------------")
        subjectId = int(input("-> Please enter the subject Id"))
        return subjectId


def cleanDirectory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def getCubeRoot(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)


def getImagePathsWithLabel(imageLabel, csvFilePath, imagesDir):
    return getImagePaths(imagesDir, getImageIdsWithLabelInputs(imageLabel, csvFilePath, imagesDir))

# def getImageIdsWithLabel(imageLabel, csvFilePath):
#     if csvFilePath is None or imageLabel is None:
#         raise ValueError("Invalid arguments")
#
#     handInfo = pd.read_csv(csvFilePath, na_filter=False)
#     return handInfo[handInfo['aspectOfHand'].str.contains(imageLabel)]['imageName'].to_numpy()


# Input: imageLabel enum inputted and the absolute path of the CSV
# Output: The imagePaths pertaining to the input label
def getImageIdsWithLabelInputs(imageLabel, csvFilePath, directoryPath):
    if csvFilePath is None or imageLabel is None:
        raise ValueError("Invalid arguments")
    handInfo = pd.read_csv(csvFilePath, na_filter=False)
    print(imageLabel, type(imageLabel), csvFilePath)
    filelist = [file for file in os.listdir(directoryPath) if file.endswith('.jpg')]
    if imageLabel == 1:
        return handInfo[handInfo['aspectOfHand'].str.contains('left') & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 2:
        return handInfo[handInfo['aspectOfHand'].str.contains('right') & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 3:
        return handInfo[handInfo['aspectOfHand'].str.contains('dorsal') & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 4:
        return handInfo[handInfo['aspectOfHand'].str.contains('palmar') & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 5:
        return handInfo[handInfo['accessories'] == 1 & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 6:
        return handInfo[handInfo['accessories'] == 0 & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 7:
        return handInfo[handInfo['gender'].str.contains('male') & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    elif imageLabel == 8:
        return handInfo[handInfo['gender'].str.contains('female') & handInfo['imageName'].isin(filelist)]['imageName'].to_numpy()
    else:
        print("imageLabel is invalid. Please try again")
        return


def getImagePaths(imagesDir, imageIds):
    if not isinstance(imageIds, list) and not isinstance(imageIds, np.ndarray):
        raise ValueError("Image Ids need to be iterable")

    imagePaths = []
    for imageId in imageIds:
        imagePaths.append(os.path.join(imagesDir, imageId))

    return imagePaths


def listFolderNames(names):
    print("-----------------------------------------------------------------------------------------")
    for i in range(len(names)):
        print(i, " > ", names[i][0])
    label = 0
    while True:
        label = int(input("-> Please enter the number: "))
        if label < len(names):
            return names[label]
        print("Invalid input, try again")


def getFolderNames(path):
    names = []
    for name in os.listdir(path):
        # print(name, os.path.abspath(name))
        names.append((name, os.path.abspath(path+name)))
    return names

def getSubjectImages(subjectId, handInfoPath):
    data = pd.read_csv("/home/worm/Desktop/ASU/CSE 515/Phase#2/MWDB/src/HandInfo.csv")
    subjectData = data[data['id'] == subjectId]
    print(subjectData.head(3))
    for image in subjectData['imageName']:
        print(image)
