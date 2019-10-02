from src.models.enums.models import ModelType
from src.dimReduction.enums.reduction import ReductionType
import os
import cv2

def getTaskFromUser():
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

        if label in ['1','2','3','4','5','6','7','8']: return label
        else: print(' Incorrect value ')

def getImagePathFromUser():
    while True:
        print("-----------------------------------------------------------------------------------------")
        imagePath = input("-> Please enter the full path of the image: ")

        if os.path.exists(imagePath): return imagePath
        else:
            print("Invalid path")

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