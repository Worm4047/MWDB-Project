import os
import cv2
import glob
import csv
import numpy as np
import datetime
from models import ColorMoments
from comparators import CMComparator

def getColorMomentsFeature(colorMomentsFeatures, imagePath):
    for colorMommentsFeature in colorMomentsFeatures:
        if colorMommentsFeature[0] == imagePath:
            return colorMommentsFeature

    raise ValueError("Image not found in database")

def getModelAndTask():
    while True:
        print("\n")
        print("-> Please the model")
        print("-> 1. Color moments")
        print("-> 2. SIFT")
        modelType = input("-> Please enter the number: ")

        if modelType == "1" or modelType == "2": break

    while True:
        print("\n")
        print("-> Please the task you would like to perform")
        print("-> 1. Describe feature vectors in human readable format")
        print("-> 2. Extract and store the feature descriptors in CSV")
        print("-> 3. Find similar images in the databases")
        taskType = input("-> Please enter the number: ")

        if taskType == "1" or taskType == "2" or taskType == "3": break

    return modelType, taskType

def getImageName():
    while True:
        print("\n")
        imageName = input("-> Please enter the file name of the image: ")

        if len(imageName) > 4: break

    return imageName

def getDistancesWithCM(imagePath):
    distances = []
    with open('featureVectorsStore/colorMomentsFeatures.csv', newline='') as csvfile:
        colorMomentsFeatures = csv.reader(csvfile, delimiter=',')
        dbImg = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
        dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)
        queryImageFeatureVector = dbImageColorMomments.featureVector

        for index, colorMommentsFeature in enumerate(colorMomentsFeatures):
            print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
            dbImagePath, imageFeatureVectorString, _ = colorMommentsFeature
            imageFeatureVectorFlat = np.array(imageFeatureVectorString.split(',')).astype(np.float)
            imageFeatureVector = imageFeatureVectorFlat.reshape((12, 16, 9))

            print("Sum: {}".format(np.sum(imageFeatureVector[:,:,6:8])))

            distance = np.linalg.norm(queryImageFeatureVector - imageFeatureVector)

            distances.append((dbImagePath, distance))

    distances.sort(key=lambda x: x[1])
    return distances

def getSimilarImages(imagePath, modelType = 1):
    if(modelType == "1"): distances = getDistancesWithCM(imagePath)

    for index, distance in enumerate(distances):
        print("ImagePath: {} , distance: {}".format(distance[0], distance[1]))

def init():
    databasePath = "/Users/yvtheja/Documents/Hands"
    modelType, taskType = getModelAndTask()
    if taskType == "1" or taskType == "3":
        imageName = getImageName()
        imagePath = os.path.join(databasePath, imageName)
        getSimilarImages(imagePath, modelType)


if __name__ == "__main__":
    init()