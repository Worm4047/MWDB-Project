import glob
import os

import pandas as pd
from scipy import spatial
from sklearn.preprocessing import normalize

from src.dimReduction.dimRedHelper import DimRedHelper
from src.dimReduction.enums.reduction import ReductionType
from src.dimReduction.latentSemantic import LatentSemantic
from src.models.enums.models import ModelType
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
from src.common.enums.labels import LabelType


# This is a temp function and is used for testing only
def findClusters(pathDorsal, pathPalmar, csvPath, inputPath):
    # inputPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/input/'
    # path = '/Users/studentworker/PycharmProjects/phase_3/test/sample/Labelled/Set1/'
    # pathDorsal = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/dorsal/'
    # pathPalmar = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/palmar/'
    # csvPath = '/Users/studentworker/PycharmProjects/phase_3/HandInfo.csv'
    # inputPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/input2/'

    reductionType = ReductionType.PCA
    modelType = ModelType.SIFT
    k = 15
    dimRedHelper = LatentSemantic()
    obj = DimRedHelper()

    all_dorsal_left_images = getLabelledDorsalLeftImages(csvPath, pathDorsal)
    all_dorsal_right_images = getLabelledDorsalRightImages(csvPath, pathDorsal)
    all_palmar_left_images = getLabelledPalmarRightImages(csvPath, pathPalmar)
    all_palmar_right_images = getLabelledPalmarLeftImages(csvPath, pathPalmar)
    inputImages = glob.glob(inputPath + "*.jpg")
    print(len(inputImages))
    print("Dorsal Left")
    dataMatrixDorsalLeft = obj.getDataMatrix(all_dorsal_left_images, ModelType.SIFT)
    # dataMatrixDorsalLeft = np.stack(dataMatrixDorsalLeft, axis=0)
    print("Dorsal Right")
    dataMatrixDorsalRight = obj.getDataMatrix(all_dorsal_right_images, ModelType.SIFT)
    # dataMatrixDorsalRight = np.stack(dataMatrixDorsalRight, axis=0)
    print("Palmar Left")
    dataMatrixPalmarLeft = obj.getDataMatrix(all_palmar_left_images, ModelType.SIFT)
    # dataMatrixPalmarLeft = np.stack(dataMatrixPalmarLeft, axis=0)
    print("Palmar Right")
    dataMatrixPalmarRight = obj.getDataMatrix(all_palmar_right_images, ModelType.SIFT)
    # dataMatrixPalmarRight = np.stack(dataMatrixPalmarRight, axis=0)
    print("Input Images")
    dataMatrixInput = obj.getDataMatrix(inputImages, ModelType.SIFT)
    # dataMatrixInput = np.stack(dataMatrixInput, axis=0)

    dorsalLeftSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixDorsalLeft, modelType,
                                                        LabelType.DORSAL, pathDorsal, all_dorsal_left_images)
    U_dorsal_left, V_dorsal_left = dorsalLeftSemantic[0], dorsalLeftSemantic[1]
    dorsalRightSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixDorsalRight, modelType,
                                                         LabelType.DORSAL, pathDorsal, all_dorsal_right_images)
    U_dorsal_right, V_dorsal_right = dorsalRightSemantic[0], dorsalRightSemantic[1]

    palmarLeftSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixPalmarLeft, modelType,
                                                        LabelType.PALMER, pathPalmar, all_palmar_left_images)
    U_palmar_left, V_palmar_left = palmarLeftSemantic[0], palmarLeftSemantic[1]
    palmarRightSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixPalmarRight, modelType,
                                                         LabelType.PALMER, pathPalmar, all_palmar_right_images)
    U_palmar_right, V_palmar_right = palmarRightSemantic[0], palmarRightSemantic[1]

    V_dorsal_left_norm = normalize(V_dorsal_left, axis=0, norm='max')
    U_dorsal_left_norm = normalize(U_dorsal_left, axis=0, norm='max')

    V_palmar_left_norm = normalize(V_palmar_left, axis=0, norm='max')
    U_palmar_left_norm = normalize(U_palmar_left, axis=0, norm='max')

    V_dorsal_right_norm = normalize(V_dorsal_right, axis=0, norm='max')
    U_dorsal_right_norm = normalize(U_dorsal_right, axis=0, norm='max')

    V_palmar_right_norm = normalize(V_palmar_right, axis=0, norm='max')
    U_palmar_right_norm = normalize(U_palmar_right, axis=0, norm='max')

    centroid_dorsal_left = np.mean(U_dorsal_left, axis=0, keepdims=True)
    centroid_dorsal_right = np.mean(U_dorsal_right, axis=0, keepdims=True)
    centroid_palmar_left = np.mean(U_palmar_left, axis=0, keepdims=True)
    centroid_palmar_right = np.mean(U_palmar_right, axis=0, keepdims=True)

    # print(centroid_dorsal.shape)
    # print(V_dorsal.shape)
    # print(U_dorsal.shape)
    # print(V_dorsal_norm.shape)
    # print(U_dorsal_norm.shape)
    print(dataMatrixInput.shape)

    dorsal_left_distances = distance.cdist(U_dorsal_left, centroid_dorsal_left, 'euclidean')
    dorsal_right_distances = distance.cdist(U_dorsal_right, centroid_dorsal_right, 'euclidean')

    palmar_left_distances = distance.cdist(U_palmar_left, centroid_palmar_left, 'euclidean')
    palmar_right_distances = distance.cdist(U_palmar_right, centroid_palmar_right, 'euclidean')

    dorsal_left_mean = np.mean(dorsal_left_distances)
    dorsal_right_mean = np.mean(dorsal_right_distances)
    palmar_left_mean = np.mean(palmar_left_distances)
    palmar_right_mean = np.mean(palmar_right_distances)

    dorsal_left_max = np.max(dorsal_left_distances)
    dorsal_right_max = np.max(dorsal_right_distances)
    palmar_left_max = np.max(palmar_left_distances)
    palmar_right_max = np.max(palmar_right_distances)

    dorsal_left_min = np.min(dorsal_left_distances)
    dorsal_right_min = np.min(dorsal_right_distances)
    palmar_left_min = np.min(palmar_left_distances)
    palmar_right_min = np.min(palmar_right_distances)

    # Using distance from centroid/mean distance from centroid
    # print(palmar_mean)
    # print(dorsal_mean)
    index = 0

    # print("*********** USING CENTROID DISTANCE *************")
    # for row in dataMatrixInput:
    #     palmar_row_reduced = np.matmul(V_palmar, row)
    #     dorsal_row_reduced = np.matmul(V_dorsal, row)
    #     palmar_dist = np.linalg.norm(palmar_row_reduced - centroid_palmar)
    #     dorsal_dist = np.linalg.norm(dorsal_row_reduced - centroid_dorsal)
    #     palmar_ratio = (palmar_dist/dorsal_max)
    #     dorsal_ratio = (dorsal_dist/palmar_max)
    #     if palmar_ratio > dorsal_ratio:
    #         print(inputImages[index]+": dorsal, dorsal:"+str(dorsal_ratio)+", palmar:"+str(palmar_ratio))
    #     elif palmar_ratio < dorsal_ratio:
    #         print(inputImages[index]+": palmar, dorsal:"+str(dorsal_ratio)+", palmar:"+str(palmar_ratio))
    #     else:
    #         print("Dorsal = Palmar")
    #     index += 1
    #
    # print("Dorsal Min:",dorsal_min)
    # print("Dorsal Max:",dorsal_max)
    # print("Palmar Min:",palmar_min)
    # print("Palmar Max:",palmar_max)

    print("*********** USING CENTROID DISTANCE *************")
    for row in dataMatrixInput:
        palmar_left_row_reduced = np.matmul(V_palmar_left, row)
        palmar_right_row_reduced = np.matmul(V_palmar_right, row)
        dorsal_left_row_reduced = np.matmul(V_dorsal_left, row)
        dorsal_right_row_reduced = np.matmul(V_dorsal_right, row)

        dorsal_left_dist = np.linalg.norm(dorsal_left_row_reduced - centroid_dorsal_left)
        dorsal_right_dist = np.linalg.norm(dorsal_right_row_reduced - centroid_dorsal_right)
        palmar_left_dist = np.linalg.norm(palmar_left_row_reduced - centroid_palmar_left)
        palmar_right_dist = np.linalg.norm(palmar_right_row_reduced - centroid_palmar_right)

        dorsal_left_ratio = (dorsal_left_dist / dorsal_left_mean)
        dorsal_right_ratio = (dorsal_right_dist / dorsal_right_mean)
        palmar_left_ratio = (palmar_left_dist / palmar_left_mean)
        palmar_right_ratio = (palmar_right_dist / palmar_right_mean)

        # dorsal_dist = np.linalg.norm(dorsal_row_reduced - centroid_dorsal)
        distanceRatios = [dorsal_left_ratio, dorsal_right_ratio, palmar_left_ratio, palmar_right_ratio]
        minRatio = min(distanceRatios)
        if distanceRatios.index(minRatio) is 0:
            print(inputImages[index] + ": DORSAL LEFT :", (' '.join(map(str, distanceRatios))))
        elif distanceRatios.index(minRatio) is 1:
            print(inputImages[index] + ": DORSAL RIGHT :", (' '.join(map(str, distanceRatios))))
        elif distanceRatios.index(minRatio) is 2:
            print(inputImages[index] + ": PALMAR LEFT :", (' '.join(map(str, distanceRatios))))
        elif distanceRatios.index(minRatio) is 3:
            print(inputImages[index] + ": PALMAR RIGHT :", (' '.join(map(str, distanceRatios))))
        index += 1

    print("*********** USING SUM *************")
    # Using SUM
    index = 0
    dorsal_images = []
    palmar_images = []
    for row in dataMatrixInput:
        total_left_dorsal = 0
        total_right_dorsal = 0
        total_left_palmar = 0
        total_right_palmar = 0

        palmar_left_row_reduced = np.matmul(V_palmar_left, row)
        palmar_right_row_reduced = np.matmul(V_palmar_right, row)
        dorsal_left_row_reduced = np.matmul(V_dorsal_left, row)
        dorsal_right_row_reduced = np.matmul(V_dorsal_right, row)

        for dorsal_row in U_dorsal_left:
            dorsal_dist = np.linalg.norm(dorsal_left_row_reduced - dorsal_row)
            total_left_dorsal += dorsal_dist
        for dorsal_row in U_dorsal_right:
            dorsal_dist = np.linalg.norm(dorsal_right_row_reduced - dorsal_row)
            total_right_dorsal += dorsal_dist
        for palmar_row in U_palmar_left:
            palmar_dist = np.linalg.norm(palmar_left_row_reduced - palmar_row)
            total_left_palmar += palmar_dist
        for palmar_row in U_palmar_right:
            palmar_dist = np.linalg.norm(palmar_right_row_reduced - palmar_row)
            total_right_palmar += palmar_dist

        distances = [total_left_dorsal, total_right_dorsal, total_left_palmar, total_right_palmar]
        minRatio = min(distances)

        if distances.index(minRatio) is 0:
            print(inputImages[index] + ": DORSAL LEFT :", (' '.join(map(str, distances))))
            dorsal_images.append(inputImages[index])
        elif distances.index(minRatio) is 1:
            print(inputImages[index] + ": DORSAL RIGHT :", (' '.join(map(str, distances))))
            dorsal_images.append(inputImages[index])
        elif distances.index(minRatio) is 2:
            print(inputImages[index] + ": PALMAR LEFT :", (' '.join(map(str, distances))))
            palmar_images.append(inputImages[index])
        elif distances.index(minRatio) is 3:
            print(inputImages[index] + ": PALMAR RIGHT :", (' '.join(map(str, distances))))
            palmar_images.append(inputImages[index])

        index += 1

    print("*********** USING MEAN *************")
    # Using Mean distance
    index = 0
    for row in dataMatrixInput:
        total_left_dorsal = 0
        total_right_dorsal = 0
        total_left_palmar = 0
        total_right_palmar = 0

        palmar_left_row_reduced = np.matmul(V_palmar_left, row)
        palmar_right_row_reduced = np.matmul(V_palmar_right, row)
        dorsal_left_row_reduced = np.matmul(V_dorsal_left, row)
        dorsal_right_row_reduced = np.matmul(V_dorsal_right, row)

        for dorsal_row in U_dorsal_left:
            dorsal_dist = np.linalg.norm(dorsal_left_row_reduced - dorsal_row)
            total_left_dorsal += dorsal_dist
        for dorsal_row in U_dorsal_right:
            dorsal_dist = np.linalg.norm(dorsal_right_row_reduced - dorsal_row)
            total_right_dorsal += dorsal_dist
        for palmar_row in U_palmar_left:
            palmar_dist = np.linalg.norm(palmar_left_row_reduced - palmar_row)
            total_left_palmar += palmar_dist
        for palmar_row in U_palmar_right:
            palmar_dist = np.linalg.norm(palmar_right_row_reduced - palmar_row)
            total_right_palmar += palmar_dist

        dist_mean_palmar_left = total_left_palmar / len(all_dorsal_left_images)
        dist_mean_palmar_right = total_right_palmar / len(all_dorsal_right_images)
        dist_mean_dorsal_left = total_left_dorsal / len(all_dorsal_left_images)
        dist_mean_dorsal_right = total_right_dorsal / len(all_dorsal_right_images)

        distances = [dist_mean_dorsal_left, dist_mean_dorsal_right, dist_mean_palmar_left, dist_mean_palmar_right]
        minRatio = min(distances)

        if distances.index(minRatio) is 0:
            print(inputImages[index] + ": DORSAL LEFT :", (' '.join(map(str, distances))))
        elif distances.index(minRatio) is 1:
            print(inputImages[index] + ": DORSAL RIGHT :", (' '.join(map(str, distances))))
        elif distances.index(minRatio) is 2:
            print(inputImages[index] + ": PALMAR LEFT :", (' '.join(map(str, distances))))
        elif distances.index(minRatio) is 3:
            print(inputImages[index] + ": PALMAR RIGHT :", (' '.join(map(str, distances))))

        index += 1

    return dorsal_images, palmar_images


# This is the main function
# Input : Datamatrix computed for the images, Image paths (absolute)
# Function :  Performs clustering of the image then visualizes them
def task1(palmarPath='/Users/studentworker/PycharmProjects/phase_3/test/sample0/palmar/',
            dorsalPath='/Users/studentworker/PycharmProjects/phase_3/test/sample0/dorsal/',
            metaDataFile='/Users/studentworker/PycharmProjects/phase_3/HandInfo.csv',
            inputPath='/Users/studentworker/PycharmProjects/phase_3/test/sample0/input2/'):
    dorsalImages, palmarImages = findClusters(palmarPath, dorsalPath, metaDataFile, inputPath)
    return dorsalImages, palmarImages


def compareWithCosine(val1, val2):
    return 1 - spatial.distance.cosine(val1, val2)


def getLabelledDorsalLeftImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, True, 0)


def getLabelledDorsalRightImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, True, 1)


def getLabelledPalmarLeftImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, False, 0)


def getLabelledPalmarRightImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, False, 1)


def getLabelledImages(csvPath, imagePath, dorsal, hand):
    label_df = pd.read_csv(csvPath)
    if dorsal and hand is 0:
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('dorsal left')]
    elif dorsal and hand is 1:
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('dorsal right')]
    elif dorsal and hand is 1:
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('palmar left')]
    else:
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('palmar right')]
    images = list(label_df['imageName'].values)
    for i in range(len(images)):
        images[i] = imagePath + images[i]
    return images


task1()
