from src.models.enums.models import ModelType
from src.dimReduction.enums.reduction import ReductionType
import os
import cv2
import pandas as pd
import numpy as np
from src.constants import DATABASE_PATH
import glob

def getImagePathsWithLabel(imageLabel, csvFilePath, imagesDir):
    return getImagePathsFromImageNames(imagesDir, getImageIdsWithLabelInputs(imageLabel, csvFilePath, imagesDir))

def getImagePathsFromImageNames(imagesDir, imageIds):
    if not isinstance(imageIds, list) and not isinstance(imageIds, np.ndarray):
        raise ValueError("Image Ids need to be iterable")

    imagePaths = []
    for imageId in imageIds:
        imagePaths.append(os.path.join(imagesDir, imageId))

    return imagePaths

def getImagePathsFromDB():
    return glob.glob(os.path.join(DATABASE_PATH, "*.jpg"))

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
