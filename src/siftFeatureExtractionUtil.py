import csv
import datetime
import glob
import os
import cv2
import numpy as np
from common.helper import getImageName, cleanDirectory

def getSIFTFeatures(imagePath):
    dbImg = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    imageGray = cv2.cvtColor(dbImg, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imageGray, None)

    return sift.detectAndCompute(imageGray, None)

def computeSIFTFeatures(databasePath):
    dbImagePaths = glob.glob(os.path.join(databasePath, "*.jpg"))
    siftStorePath = "featureVectorsStore/siftStore"
    cleanDirectory(siftStorePath)
    for index, dbImagePath in enumerate(dbImagePaths):
        print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
        keyPoints, descriptors = getSIFTFeatures(dbImagePath)

        deserialisedKeyPoints = []
        for keyPoint in keyPoints:
            deserialisedKeyPoints.append([keyPoint.pt[0], keyPoint.pt[1], keyPoint.size, keyPoint.angle])

        imageFileName = getImageName(dbImagePath)
        imageName = imageFileName.split('.')[0]

        np.save(os.path.join(siftStorePath, "{}_des".format(imageName)), descriptors)
        np.save(os.path.join(siftStorePath, "{}_kp".format(imageName)), np.array(deserialisedKeyPoints))

def getCosineSimilarity(des1, des2):
    dot = np.dot(des1, des2)
    norma = np.linalg.norm(des1)
    normb = np.linalg.norm(des2)
    return dot / (norma * normb)

def getEcludianDistance(des1, des2):
    return np.linalg.norm(des1 - des2)

def getDesSimilarity(queryDes, des):
    return getCosineSimilarity(queryDes, des)

def getDistances(queryDes, queryDesIndex, desList, distancesTable):
    for listIndex, des in enumerate(desList):
        distancesTable.append(("{}_{}".format(queryDesIndex, listIndex), getEcludianDistance(queryDes, des)))

def getSIFTDistance(aDes, bDes):
    primaryDes = []
    secondaryDes = []

    if(aDes.shape[0] > bDes.shape[1]):
        primaryDes = bDes
        secondaryDes = aDes
    else:
        primaryDes = aDes
        secondaryDes = bDes

    distancesTable = []
    for index, des in enumerate(primaryDes):
        getDistances(des, index, secondaryDes, distancesTable)

    distancesTable.sort(key=lambda x: x[1])
    primaryDesOccupancy = {}
    secondaryDesOccupancy = {}
    totalDistance = 0


    count = 0
    for desDistance in distancesTable:
        if(count > 30): break

        key, distance = desDistance
        primaryDesIndex, secondaryDesIndex = key.split("_")

        if primaryDesIndex in primaryDesOccupancy or secondaryDesIndex in secondaryDesOccupancy: continue

        primaryDesOccupancy[primaryDesIndex] = primaryDesIndex
        secondaryDesOccupancy[secondaryDesIndex] = secondaryDesIndex
        totalDistance += distance
        count += 1

    return totalDistance/count


