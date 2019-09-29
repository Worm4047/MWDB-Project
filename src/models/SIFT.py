from src.models.interfaces.Model import Model
import cv2
import numpy as np

class SIFT(Model):
    def __init__(self, grayScaleImage):
        self.validateImage(grayScaleImage)
        super(SIFT, self).__init__()
        self.keypoints, self.descriptors = self.getSIFTFeatures(grayScaleImage)
        self.deserialisedKeyPoints = []
        self.deserialiseKeyPoints()

    def getFeatures(self):
        return self.descriptors

    def validateImage(self, image):
        if image.shape is None:
            raise ValueError("Not a np array")
        if len(image.shape) != 2:
            raise ValueError("Invalid Image")

    def deserialiseKeyPoints(self):
        for keyPoint in self.keypoints:
            self.deserialisedKeyPoints.append([keyPoint.pt[0], keyPoint.pt[1], keyPoint.size, keyPoint.angle])

    def getSIFTFeatures(self, imageGray):
        if imageGray is None:
            raise ValueError("Image can not be None")

        sift = cv2.xfeatures2d.SIFT_create()
        return sift.detectAndCompute(imageGray, None)

    def compare(self, siftModel):
        if not isinstance(siftModel, SIFT):
            raise ValueError("Not a SIFT model")

        return self.getSIFTDistance(self.descriptors, siftModel.getFeatures())

    def getDistances(self, queryDes, queryDesIndex, desList, distancesTable):
        for listIndex, des in enumerate(desList):
            distancesTable.append(("{}_{}".format(queryDesIndex, listIndex), np.linalg.norm(queryDes - des)))

    def getSIFTDistance(self, aDes, bDes):
        primaryDes = []
        secondaryDes = []

        if (aDes.shape[0] > bDes.shape[1]):
            primaryDes = bDes
            secondaryDes = aDes
        else:
            primaryDes = aDes
            secondaryDes = bDes

        distancesTable = []
        for index, des in enumerate(primaryDes):
            self.getDistances(des, index, secondaryDes, distancesTable)

        distancesTable.sort(key=lambda x: x[1])
        primaryDesOccupancy = {}
        secondaryDesOccupancy = {}
        totalDistance = 0

        count = 0
        for desDistance in distancesTable:
            if (count > 30): break

            key, distance = desDistance
            primaryDesIndex, secondaryDesIndex = key.split("_")

            if primaryDesIndex in primaryDesOccupancy or secondaryDesIndex in secondaryDesOccupancy: continue

            primaryDesOccupancy[primaryDesIndex] = primaryDesIndex
            secondaryDesOccupancy[secondaryDesIndex] = secondaryDesIndex
            totalDistance += distance
            count += 1

        return totalDistance / count

