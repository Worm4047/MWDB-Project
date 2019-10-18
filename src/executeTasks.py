import cv2

from src.common.imageFeatureHelper import getImageFeatures
from src.common.latentSemanticsHelper import getSemanticsFromFolder, getParams
from src.dimReduction.dimRedHelper import getQueryImageRep
from src.models.enums.models import ModelType
from src.task8 import initTask8
from src.dimReduction import dimRedHelper
from src.common import dataMatrixHelper, comparisonHelper
from src.common import util
from src.common import helper

#need to test
def task1(directoryPath, modelType, k, dimRecTechnique):
    print(" EXECUTING TASK 1 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    data_matrix = dimRedHelper.getDataMatrix(None, modelType, directoryPath)
    latent_semantic = dimRedHelper.getLatentSemantic(k, dimRecTechnique, data_matrix, modelType, None, directoryPath)
    print("In terms of data")
    twpairData = util.sort_print_n_return(latent_semantic[0])
    util.visualize_ec(twpairData, "data", latent_semantic[0])
    print("In terms of feature")
    twpairFeat = util.sort_print_n_return(latent_semantic[1])
    util.visualize_ec(twpairFeat, "feature", latent_semantic[1])



def task2(foldername, folderPath, imagePath, m):
    # call a function to get data from folder name
    print(" EXECUTING TASK 2 ")
    print(folderPath)
    print(imagePath)
    U, V = getSemanticsFromFolder(folderPath)
    dir, modelType, dimRidTechnique, K, label = getParams(foldername)
    query_image_features = getQueryImageRep(V, imagePath, modelType)
    list = comparisonHelper.getMSimilarImages(U, query_image_features, m, modelType)


#need to test
def task3(directoryPath, modelType, k, dimRecTechnique, label):
    print(" EXECUTING TASK 3 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    print(label)
    data_matrix = dimRedHelper.getDataMatrix(None, modelType, directoryPath)
    images_list_with_label = helper.getImageIdsWithLabelInputs(label, csv_path)
    import os
    all_images = []
    for file in os.listdir(directoryPath):
        if file.endswith('.jpg'):
            all_images.append(file[:-4])
    data_matrix_by_label = dataMatrixHelper.filter_by_label(data_matrix, all_images, images_list_with_label)
    latent_semantic = dimRedHelper.getLatentSemantic(k, dimRecTechnique, data_matrix_by_label, modelType, label, directoryPath)
    print("In terms of data")
    twpairData = util.sort_print_n_return(latent_semantic[0])
    util.visualize_ec(twpairData, "data", latent_semantic[0])
    print("In terms of feature")
    twpairFeat = util.sort_print_n_return(latent_semantic[1])
    util.visualize_ec(twpairFeat, "feature", latent_semantic[1])


def task4(foldername, folderPath, imagePath, m):
    print(" EXECUTING TASK 4 ")
    print(folderPath)
    print(imagePath)


def task5(foldername, folderPath, imagePath):
    print(" EXECUTING TASK 5 ")
    print(folderPath)
    print(imagePath)


def task8(imageDir, handInfoCSV, k):
    initTask8(imageDir, handInfoCSV, k)
    print(" EXECUTING TASK 8 ")