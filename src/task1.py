import glob
import os

import pandas as pd
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
def helper():
    # pathDorsal = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/dorsal/'
    # pathPalmar = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/palmar/'
    # inputPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/input/'

    path = '/Users/studentworker/PycharmProjects/phase_3/test/sample/Labelled/Set1/'
    csvPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample/labelled_set1.csv'
    inputPath = '/Users/studentworker/PycharmProjects/phase_3/test/sample0/input/'
    all_dorsal_images = getLabelledDorsalImages(csvPath, path)
    all_palmar_images = getLabelledPalmarImages(csvPath, path)
    inputImages = glob.glob(inputPath + "*.jpg")
    obj = DimRedHelper()
    dataMatrixDorsal = obj.getDataMatrixForLBP(all_dorsal_images, [])
    dataMatrixDorsal = np.stack(dataMatrixDorsal, axis=0)
    dataMatrixPalmar = obj.getDataMatrixForLBP(all_palmar_images, [])
    dataMatrixPalmar = np.stack(dataMatrixPalmar, axis=0)
    dataMatrixInput = obj.getDataMatrixForLBP(inputImages, [])
    dataMatrixInput = np.stack(dataMatrixInput, axis=0)

    reductionType = ReductionType.NMF
    modelType = ModelType.LBP
    k = 40

    dimRedHelper = LatentSemantic()
    dorsalSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixDorsal, modelType, LabelType.DORSAL,
                                                        path, all_dorsal_images)
    U_dorsal, V_dorsal = dorsalSemantic
    palmarSemantic = dimRedHelper.getLatentSemantic(k, reductionType, dataMatrixPalmar, modelType, LabelType.PALMER,
                                                        path, all_palmar_images)
    U_palmar, V_palmar  = palmarSemantic

    V_dorsal_norm = normalize(V_dorsal, axis=0, norm='max')
    V_palmar_norm = normalize(V_palmar, axis=0, norm='max')

    centroid_dorsal = np.mean(V_dorsal_norm, axis=0, keepdims=True)
    centroid_palmar = np.mean(V_palmar_norm, axis=0, keepdims=True)

    dorsal_distances = distance.cdist(V_dorsal_norm, centroid_dorsal, 'euclidean')
    palmar_distances = distance.cdist(V_palmar_norm, centroid_palmar, 'euclidean')

    dorsal_mean = np.mean(dorsal_distances)
    palmar_mean = np.mean(palmar_distances)

    print(palmar_mean)
    print(dorsal_mean)
    index = 0
    for row in dataMatrixInput:
        palmar_dist = np.linalg.norm(row - centroid_palmar)
        dorsal_dist = np.linalg.norm(row - centroid_dorsal)
        palmar_ratio = (palmar_dist/palmar_mean)
        dorsal_ratio = (dorsal_dist/dorsal_mean)
        print(palmar_dist)
        print(dorsal_dist)
        print(palmar_ratio)
        print(dorsal_ratio)
        if palmar_ratio > dorsal_ratio:
            print(inputImages[index]+": dorsal")
        elif palmar_ratio < dorsal_ratio:
            print(inputImages[index] + ": palmar")
        else:
            print("Dorsal = Palmar")
        index += 1


# This is the main function
# Input : Datamatrix computed for the images, Image paths (absolute)
# Function :  Performs clustering of the image then visualizes them
def task1(dm, images, c=10):
    print(dm.shape, len(images))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(dm)
    clusters = kmeans.cluster_centers_
    for cluster in clusters:
        print(cluster.shape)


def getLabelledDorsalImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, True)


def getLabelledPalmarImages(csvpath, imagePath):
    return getLabelledImages(csvpath, imagePath, False)


def getLabelledImages(csvPath, imagePath, dorsal):
    label_df = pd.read_csv(csvPath)
    if dorsal:
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('dorsal')]
    else:
        label_df = label_df.loc[label_df['aspectOfHand'].str.contains('palmar')]
    images = list(label_df['imageName'].values)
    for i in range(len(images)):
        images[i] = imagePath + images[i]
    return images


helper()
