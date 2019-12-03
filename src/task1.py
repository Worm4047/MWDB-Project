import glob
import os
import ntpath
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
def classifyImages(imagesFolder, csvPath, testPath, metaDataTest):
    # testPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/input/'
    # path = '/Users/studentworker/PycharmProjects/phase_3/test/sample/Labelled/Set1/'
    # pathDorsal = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/dorsal/'
    # pathPalmar = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/palmar/'
    # csvPath = '/Users/studentworker/PycharmProjects/phase_3/HandInfo.csv'
    # testPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/input2/'

    reductionType = ReductionType.SVD
    modelType = ModelType.HOG
    k = 15
    dimRedHelper = LatentSemantic()
    obj = DimRedHelper()

    all_dorsal_left_images = getLabelledDorsalLeftImages(csvPath, imagesFolder)
    all_dorsal_right_images = getLabelledDorsalRightImages(csvPath, imagesFolder)
    all_palmar_right_images = getLabelledPalmarRightImages(csvPath, imagesFolder)
    all_palmar_left_images = getLabelledPalmarLeftImages(csvPath, imagesFolder)

    if len(all_dorsal_left_images) < k:
        all_dorsal_right_images += all_dorsal_left_images
        all_dorsal_left_images = None

    if len(all_dorsal_right_images) < k:
        all_dorsal_left_images += all_dorsal_right_images
        all_dorsal_right_images = None

    if len(all_palmar_left_images) < k:
        all_palmar_right_images += all_palmar_left_images
        all_palmar_left_images = None

    if len(all_palmar_right_images) < k:
        all_palmar_left_images += all_palmar_right_images
        all_palmar_right_images = None


    testImages = glob.glob(testPath + "*.jpg")
    print(len(testImages))

    dataMatrixDorsalLeft = None
    dataMatrixDorsalRight = None
    dataMatrixPalmarLeft = None
    dataMatrixPalmarRight = None

    if(all_dorsal_left_images != None):
        print("Dorsal Left")
        dataMatrixDorsalLeft = obj.getDataMatrix(all_dorsal_left_images, ModelType.HOG)
        # dataMatrixDorsalLeft = np.stack(dataMatrixDorsalLeft, axis=0)

    if (all_dorsal_right_images != None):
        print("Dorsal Right")
        dataMatrixDorsalRight = obj.getDataMatrix(all_dorsal_right_images, ModelType.HOG)
        # dataMatrixDorsalRight = np.stack(dataMatrixDorsalRight, axis=0)

    if (all_palmar_left_images != None):
        print("Palmar Left")
        dataMatrixPalmarLeft = obj.getDataMatrix(all_palmar_left_images, ModelType.HOG)
        # dataMatrixPalmarLeft = np.stack(dataMatrixPalmarLeft, axis=0)

    if (all_palmar_right_images != None):
        print("Palmar Right")
        dataMatrixPalmarRight = obj.getDataMatrix(all_palmar_right_images, ModelType.HOG)
        # dataMatrixPalmarRight = np.stack(dataMatrixPalmarRight, axis=0)

    print("Input Images")
    dataMatrixTest = obj.getDataMatrix(testImages, ModelType.HOG)
    # dataMatrixInput = np.stack(dataMatrixInput, axis=0)

    U_dorsal_left = None
    V_dorsal_left = None
    U_dorsal_right = None
    V_dorsal_right = None
    U_palmar_left = None
    V_palmar_left = None
    U_palmar_right = None
    V_palmar_right = None

    if dataMatrixDorsalLeft is not None:
        dorsalLeftSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixDorsalLeft, modelType,
                                                            LabelType.DORSAL, imagesFolder, all_dorsal_left_images)
        U_dorsal_left, V_dorsal_left = dorsalLeftSemantic[0], dorsalLeftSemantic[1]

    if dataMatrixDorsalRight is not None:
        dorsalRightSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixDorsalRight, modelType,
                                                             LabelType.DORSAL, imagesFolder, all_dorsal_right_images)
        U_dorsal_right, V_dorsal_right = dorsalRightSemantic[0], dorsalRightSemantic[1]

    if dataMatrixPalmarLeft is not None:
        palmarLeftSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixPalmarLeft, modelType,
                                                            LabelType.PALMER, imagesFolder, all_palmar_left_images)
        U_palmar_left, V_palmar_left = palmarLeftSemantic[0], palmarLeftSemantic[1]

    if dataMatrixPalmarRight is not None:
        palmarRightSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixPalmarRight, modelType,
                                                             LabelType.PALMER, imagesFolder, all_palmar_right_images)
        U_palmar_right, V_palmar_right = palmarRightSemantic[0], palmarRightSemantic[1]

    if V_dorsal_left is not None and U_dorsal_left is not None:
        V_dorsal_left_norm = normalize(V_dorsal_left, axis=0, norm='max')
        U_dorsal_left_norm = normalize(U_dorsal_left, axis=0, norm='max')

    if V_palmar_left is not None and U_palmar_left is not None:
        V_palmar_left_norm = normalize(V_palmar_left, axis=0, norm='max')
        U_palmar_left_norm = normalize(U_palmar_left, axis=0, norm='max')

    if V_dorsal_right is not None and U_dorsal_right is not None:
        V_dorsal_right_norm = normalize(V_dorsal_right, axis=0, norm='max')
        U_dorsal_right_norm = normalize(U_dorsal_right, axis=0, norm='max')

    if V_palmar_right is not None and U_palmar_right is not None:
        V_palmar_right_norm = normalize(V_palmar_right, axis=0, norm='max')
        U_palmar_right_norm = normalize(U_palmar_right, axis=0, norm='max')

    dorsal_left_mean = None
    dorsal_right_mean = None
    palmar_left_mean = None
    palmar_right_mean = None

    centroid_dorsal_left = None
    centroid_dorsal_right = None
    centroid_palmar_left = None
    centroid_palmar_right = None

    if U_dorsal_left is not None:
        centroid_dorsal_left = np.mean(U_dorsal_left, axis=0, keepdims=True)
        dorsal_left_distances = distance.cdist(U_dorsal_left, centroid_dorsal_left, 'euclidean')
        dorsal_left_mean = np.mean(dorsal_left_distances)
        dorsal_left_max = np.max(dorsal_left_distances)
        dorsal_left_min = np.min(dorsal_left_distances)
    if U_dorsal_right is not None:
        centroid_dorsal_right = np.mean(U_dorsal_right, axis=0, keepdims=True)
        dorsal_right_distances = distance.cdist(U_dorsal_right, centroid_dorsal_right, 'euclidean')
        dorsal_right_mean = np.mean(dorsal_right_distances)
        dorsal_right_max = np.max(dorsal_right_distances)
        dorsal_right_min = np.min(dorsal_right_distances)
    if U_palmar_left is not None:
        centroid_palmar_left = np.mean(U_palmar_left, axis=0, keepdims=True)
        palmar_left_distances = distance.cdist(U_palmar_left, centroid_palmar_left, 'euclidean')
        palmar_left_mean = np.mean(palmar_left_distances)
        palmar_left_max = np.max(palmar_left_distances)
        palmar_left_min = np.min(palmar_left_distances)
    if U_palmar_right is not None:
        centroid_palmar_right = np.mean(U_palmar_right, axis=0, keepdims=True)
        palmar_right_distances = distance.cdist(U_palmar_right, centroid_palmar_right, 'euclidean')
        palmar_right_mean = np.mean(palmar_right_distances)
        palmar_right_max = np.max(palmar_right_distances)
        palmar_right_min = np.min(palmar_right_distances)

    # print(centroid_dorsal.shape)
    # print(V_dorsal.shape)
    # print(U_dorsal.shape)
    # print(V_dorsal_norm.shape)
    # print(U_dorsal_norm.shape)
    print(dataMatrixTest.shape)

    # Using distance from centroid/mean distance from centroid
    # print(palmar_mean)
    # print(dorsal_mean)
    index = 0

    dorsal_images = []
    palmar_images = []
    print("*********** USING CENTROID DISTANCE *************")
    for row in dataMatrixTest:
        dorsal_left_ratio = 999999
        dorsal_right_ratio = 999999
        palmar_left_ratio = 999999
        palmar_right_ratio = 999999
        if V_palmar_left is not None:
            palmar_left_row_reduced = np.matmul(V_palmar_left, row)
            palmar_left_dist = np.linalg.norm(palmar_left_row_reduced - centroid_palmar_left)
            palmar_left_ratio = (palmar_left_dist / palmar_left_mean)
        if V_palmar_right is not None:
            palmar_right_row_reduced = np.matmul(V_palmar_right, row)
            palmar_right_dist = np.linalg.norm(palmar_right_row_reduced - centroid_palmar_right)
            palmar_right_ratio = (palmar_right_dist / palmar_right_mean)
        if V_dorsal_left is not None:
            dorsal_left_row_reduced = np.matmul(V_dorsal_left, row)
            dorsal_left_dist = np.linalg.norm(dorsal_left_row_reduced - centroid_dorsal_left)
            dorsal_left_ratio = (dorsal_left_dist / dorsal_left_mean)
        if V_dorsal_right is not None:
            dorsal_right_row_reduced = np.matmul(V_dorsal_right, row)
            dorsal_right_dist = np.linalg.norm(dorsal_right_row_reduced - centroid_dorsal_right)
            dorsal_right_ratio = (dorsal_right_dist / dorsal_right_mean)

        # dorsal_dist = np.linalg.norm(dorsal_row_reduced - centroid_dorsal)
        distanceRatios = [dorsal_left_ratio, dorsal_right_ratio, palmar_left_ratio, palmar_right_ratio]
        minRatio = min(distanceRatios)
        if distanceRatios.index(minRatio) is 0:
            print(testImages[index] + ": PALMAR LEFT :", (' '.join(map(str, distanceRatios))))
            palmar_images.append(testImages[index])
        elif distanceRatios.index(minRatio) is 1:
            print(testImages[index] + ": PALMAR RIGHT :", (' '.join(map(str, distanceRatios))))
            palmar_images.append(testImages[index])
        elif distanceRatios.index(minRatio) is 2:
            print(testImages[index] + ": DORSAL LEFT :", (' '.join(map(str, distanceRatios))))
            dorsal_images.append(testImages[index])
        elif distanceRatios.index(minRatio) is 3:
            print(testImages[index] + ": DORSAL RIGHT :", (' '.join(map(str, distanceRatios))))
            dorsal_images.append(testImages[index])
        index += 1

    accuracy = calculateAccuracy(dorsal_images, palmar_images, metaDataTest)
    return dorsal_images, palmar_images, accuracy

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



    # print("*********** USING SUM *************")
    # # Using SUM
    # index = 0
    # dorsal_images = []
    # palmar_images = []
    # for row in dataMatrixTest:
    #     total_left_dorsal = 0
    #     total_right_dorsal = 0
    #     total_left_palmar = 0
    #     total_right_palmar = 0
    #
    #     palmar_left_row_reduced = np.matmul(V_palmar_left, row)
    #     palmar_right_row_reduced = np.matmul(V_palmar_right, row)
    #     dorsal_left_row_reduced = np.matmul(V_dorsal_left, row)
    #     dorsal_right_row_reduced = np.matmul(V_dorsal_right, row)
    #
    #     for dorsal_row in U_dorsal_left:
    #         dorsal_dist = np.linalg.norm(dorsal_left_row_reduced - dorsal_row)
    #         total_left_dorsal += dorsal_dist
    #     for dorsal_row in U_dorsal_right:
    #         dorsal_dist = np.linalg.norm(dorsal_right_row_reduced - dorsal_row)
    #         total_right_dorsal += dorsal_dist
    #     for palmar_row in U_palmar_left:
    #         palmar_dist = np.linalg.norm(palmar_left_row_reduced - palmar_row)
    #         total_left_palmar += palmar_dist
    #     for palmar_row in U_palmar_right:
    #         palmar_dist = np.linalg.norm(palmar_right_row_reduced - palmar_row)
    #         total_right_palmar += palmar_dist
    #
    #     distances = [total_left_dorsal, total_right_dorsal, total_left_palmar, total_right_palmar]
    #     minRatio = min(distances)
    #
    #     if distances.index(minRatio) is 0:
    #         print(testImages[index] + ": DORSAL LEFT :", (' '.join(map(str, distances))))
    #         dorsal_images.append(testImages[index])
    #     elif distances.index(minRatio) is 1:
    #         print(testImages[index] + ": DORSAL RIGHT :", (' '.join(map(str, distances))))
    #         dorsal_images.append(testImages[index])
    #     elif distances.index(minRatio) is 2:
    #         print(testImages[index] + ": PALMAR LEFT :", (' '.join(map(str, distances))))
    #         palmar_images.append(testImages[index])
    #     elif distances.index(minRatio) is 3:
    #         print(testImages[index] + ": PALMAR RIGHT :", (' '.join(map(str, distances))))
    #         palmar_images.append(testImages[index])
    #
    #     index += 1
    #
    # print("*********** USING MEAN *************")
    # # Using Mean distance
    # index = 0
    # for row in dataMatrixTest:
    #     total_left_dorsal = 0
    #     total_right_dorsal = 0
    #     total_left_palmar = 0
    #     total_right_palmar = 0
    #
    #     palmar_left_row_reduced = np.matmul(V_palmar_left, row)
    #     palmar_right_row_reduced = np.matmul(V_palmar_right, row)
    #     dorsal_left_row_reduced = np.matmul(V_dorsal_left, row)
    #     dorsal_right_row_reduced = np.matmul(V_dorsal_right, row)
    #
    #     for dorsal_row in U_dorsal_left:
    #         dorsal_dist = np.linalg.norm(dorsal_left_row_reduced - dorsal_row)
    #         total_left_dorsal += dorsal_dist
    #     for dorsal_row in U_dorsal_right:
    #         dorsal_dist = np.linalg.norm(dorsal_right_row_reduced - dorsal_row)
    #         total_right_dorsal += dorsal_dist
    #     for palmar_row in U_palmar_left:
    #         palmar_dist = np.linalg.norm(palmar_left_row_reduced - palmar_row)
    #         total_left_palmar += palmar_dist
    #     for palmar_row in U_palmar_right:
    #         palmar_dist = np.linalg.norm(palmar_right_row_reduced - palmar_row)
    #         total_right_palmar += palmar_dist
    #
    #     dist_mean_palmar_left = total_left_palmar / len(all_dorsal_left_images)
    #     dist_mean_palmar_right = total_right_palmar / len(all_dorsal_right_images)
    #     dist_mean_dorsal_left = total_left_dorsal / len(all_dorsal_left_images)
    #     dist_mean_dorsal_right = total_right_dorsal / len(all_dorsal_right_images)
    #
    #     distances = [dist_mean_dorsal_left, dist_mean_dorsal_right, dist_mean_palmar_left, dist_mean_palmar_right]
    #     minRatio = min(distances)
    #
    #     if distances.index(minRatio) is 0:
    #         print(testImages[index] + ": DORSAL LEFT :", (' '.join(map(str, distances))))
    #     elif distances.index(minRatio) is 1:
    #         print(testImages[index] + ": DORSAL RIGHT :", (' '.join(map(str, distances))))
    #     elif distances.index(minRatio) is 2:
    #         print(testImages[index] + ": PALMAR LEFT :", (' '.join(map(str, distances))))
    #     elif distances.index(minRatio) is 3:
    #         print(testImages[index] + ": PALMAR RIGHT :", (' '.join(map(str, distances))))
    #
    #     index += 1

# This is the main function
# Input : Datamatrix computed for the images, Image paths (absolute)
# Function :  Performs clustering of the image then visualizes them

# def task1(imagesFolder='/Users/studentworker/PycharmProjects/phase_3/test/sample0/input/',
#           metaDataFile='/Users/studentworker/PycharmProjects/phase_3/HandInfo.csv',
#           testPath='/Users/studentworker/PycharmProjects/phase_3/test/sample0/test/'):

def task1(imagesFolder='static/Labelled/Set2/',
          metaDataFile='static/labelled_set2.csv',
          testPath='static/Unlabelled/Set1/',
          metaDataTestFile='static/HandInfo.csv'):
    dorsalImages, palmarImages, accuracy = classifyImages(imagesFolder, metaDataFile, testPath, metaDataTestFile)
    return dorsalImages, palmarImages, accuracy


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
    print("Dorsal:",dorsal,"left:",hand)
    if dorsal and hand is 0:
        print("11111111")
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('dorsal left')]
    elif dorsal and hand is 1:
        print("22222222")
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('dorsal right')]
    elif not dorsal and hand is 0:
        print("33333333")
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('palmar left')]
    else:
        print("44444444")
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('palmar right')]
    images = list(label_df['imageName'].values)
    for i in range(len(images)):
        images[i] = imagePath + images[i]
    return images

def calculateAccuracy(dorsal_images, palmar_images, csvPath):
    label_df = pd.read_csv(csvPath)
    accuracy = 0
    for image in dorsal_images:
        name = path_leaf(image)
        actual_label = label_df.at[label_df['imageName'].eq(name).idxmax(), 'aspectOfHand']
        print(name, " - ", actual_label, ":dorsal")
        if 'dorsal' in actual_label:
            accuracy += 1
    for image in palmar_images:
        name = path_leaf(image)
        actual_label = label_df.at[label_df['imageName'].eq(name).idxmax(), 'aspectOfHand']
        print(name, " - ", actual_label, ":palmar")
        if 'palmar' in actual_label:
            accuracy += 1
    print("Accuracy:", accuracy/(len(dorsal_images)+len(palmar_images)))


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# task1()
