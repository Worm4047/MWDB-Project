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
            # print(colorMommentsFeature[0])
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

    # if taskType == "3":
    #     while True:
    #         print("\n")
    #         imagePath = input("-> Please enter the absolute image path of the query image: ")

def getImageName():
    while True:
        print("\n")
        imageName = input("-> Please enter the file name of the image: ")

        if len(imageName) > 4: break

    return imageName

def getDistancesWithCM(imagePath):
    distances = []
    with open('colorMomentsFeatures.csv', newline='') as csvfile:
        colorMomentsFeatures = csv.reader(csvfile, delimiter=',')
        # _, queryImageMeanString, queryImageVarianceString, queryImageSkewString = getColorMomentsFeature(
        #     colorMomentsFeatures, imagePath)
        # queryImageMean = np.array(queryImageMeanString.split(',')).astype(np.float)
        # queryImageVariance = np.array(queryImageVarianceString.split(',')).astype(np.float)
        # queryImageSkew = np.array(queryImageSkewString.split(',')).astype(np.float)

        dbImg = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
        dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)
        queryImageMean = dbImageColorMomments.meanFeatureVector.flatten()
        queryImageVariance = dbImageColorMomments.varianceFeatureVector.flatten()
        queryImageSkew = dbImageColorMomments.skewFeatureVector.flatten()
        queryImageFeatureVector = dbImageColorMomments.featureVector

        for index, colorMommentsFeature in enumerate(colorMomentsFeatures):
            distance = 0
            print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
            dbImagePath, imageMeanString, imageVarianceString, imageSkewString = colorMommentsFeature
            imageMean = np.array(imageMeanString.split(',')).astype(np.float)
            imageVariance = np.array(imageVarianceString.split(',')).astype(np.float)
            imageSkew = np.array(imageSkewString.split(',')).astype(np.float)

            distance += CMComparator.CMComparator().compare(queryImageMean, imageMean)
            distance += CMComparator.CMComparator().compare(queryImageVariance, imageVariance)
            distance += CMComparator.CMComparator().compare(queryImageSkew, imageSkew)

            distances.append((dbImagePath, distance / 3))

    distances.sort(key=lambda x: x[1])
    return distances

def getSimilarImages(imagePath, modelType = 1):
    if(modelType == "1"): distances = getDistancesWithCM(imagePath)

    for index, distance in enumerate(distances):
        print("ImagePath: {} , distance: {}".format(distance[0], distance[1]))

def init():
    databasePath = "/Users/yvtheja/Downloads/Hands"
    modelType, taskType = getModelAndTask()
    if taskType == "1" or taskType == "3":
        imageName = getImageName()
        imagePath = os.path.join(databasePath, imageName)
        getSimilarImages(imagePath, modelType)


if __name__ == "__main__":
    init()


    # for index, dbImagePath in enumerate(dbImagePaths):
    #     print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
    #     distance = 0
    #     dbImg = cv2.imread(dbImagePath, cv2.IMREAD_COLOR)
    #     print("Time: {} | Processing: {} | Image read".format(datetime.datetime.now(), index))
    #     dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
    #     print("Time: {} | Processing: {} | To YUV".format(datetime.datetime.now(), index))
    #     dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)
    #     print("Time: {} | Processing: {} | Compute color moments".format(datetime.datetime.now(), index))
    #     distance += CMComparator.CMComparator().compare(colorMomments.meanFeatureVector, dbImageColorMomments.meanFeatureVector)
    #     print("Time: {} | Processing: {} | Calculate Ecludian".format(datetime.datetime.now(), index))
    #     distance += CMComparator.CMComparator().compare(colorMomments.varianceFeatureVector,
    #                                         dbImageColorMomments.varianceFeatureVector)
    #     print("Time: {} | Processing: {} | Calculate Ecludian".format(datetime.datetime.now(), index))
    #     distance += CMComparator.CMComparator().compare(colorMomments.skewFeatureVector,
    #                                         dbImageColorMomments.skewFeatureVector)
    #     print("Time: {} | Processing: {} | Calculate Ecludian".format(datetime.datetime.now(), index))
    #
    #     distances.append((dbImagePath, distance/3))




    # imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # keyPoints, descriptors = sift.detectAndCompute(imageGray, None)
    #
    # print("Keypoints shape: {} | Descriptors shape: {}".format(len(keyPoints), descriptors.shape))
