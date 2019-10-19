from src.common import dataMatrixHelper, comparisonHelper
from src.common import helper
from src.common import util
from src.common.latentSemanticsHelper import getSemanticsFromFolder, getParams
from src.common.plotHelper import plotFigures
from src.dimReduction import dimRedHelper
from src.dimReduction.dimRedHelper import getQueryImageRep
from src.task5 import initTask5
from src.task8 import initTask8

import os
from src.common.latentSemanticsHelper import getLatentSemanticPath
from src.task5 import initTask5

def task1(directoryPath, modelType, k, dimRecTechnique):
    print(" EXECUTING TASK 1 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    all_images = []
    for file in os.listdir(directoryPath):
        if file.endswith(".jpg"):
            all_images.append(file)
    image_paths = [os.path.join(directoryPath, "{}".format(imagename)) for imagename in all_images]
    data_matrix = dimRedHelper.getDataMatrix(image_paths, modelType, None, directoryPath)
    latent_semantic = dimRedHelper.getLatentSemantic(k, dimRecTechnique, data_matrix, modelType, None, os.path.basename(directoryPath), image_paths)
    print("In terms of data")
    twpairData = util.sort_print_n_return(latent_semantic[0].transpose())
    # util.visualize_ec(twpairData, "data", None, directoryPath, all_images)
    print("In terms of feature")
    twpairFeat = util.sort_print_n_return(latent_semantic[1])
    util.visualize_ec(twpairFeat, "feature", data_matrix, directoryPath, all_images)

def task2(foldername, folderPath, imagePath, m):
    # call a function to get data from folder name
    print(" EXECUTING TASK 2 ")
    print(folderPath)
    print(imagePath)
    U, V, imagePaths = getSemanticsFromFolder(folderPath)
    dir, modelType, dimRidTechnique, K, label = getParams(foldername)
    query_image_features = getQueryImageRep(V, imagePath, modelType)
    list = comparisonHelper.getMSimilarImages(U, query_image_features, m, imagePaths)
    plotFigures(list)

def task3(directoryPath, modelType, k, dimRecTechnique, label):
    print(" EXECUTING TASK 3 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    print(label)
    images_list_with_label = ["{}.jpg".format(imagename) for imagename in helper.getImageIdsWithLabelInputs(label, "./store/HandInfo.csv")]
    image_paths = [os.path.join(directoryPath, "{}".format(imagename)) for imagename in images_list_with_label]
    data_matrix = dimRedHelper.getDataMatrix(image_paths, modelType, label, directoryPath)
    latent_semantic = dimRedHelper.getLatentSemantic(k, dimRecTechnique, data_matrix, modelType, None, os.path.basename(directoryPath),
                                   image_paths)
    print("In terms of data")
    twpairData = util.sort_print_n_return(latent_semantic[0].transpose())
    util.visualize_ec(twpairData, "data", data_matrix, directoryPath, images_list_with_label)
    print("In terms of feature")
    twpairFeat = util.sort_print_n_return(latent_semantic[1])
    util.visualize_ec(twpairFeat, "feature", data_matrix, directoryPath, images_list_with_label)


def task4(foldername, folderPath, imagePath, m):
    print(" EXECUTING TASK 4 ")
    print(folderPath)
    print(imagePath)
    U, V, imagePaths = getSemanticsFromFolder(folderPath)
    dir, modelType, dimRidTechnique, K, label = getParams(foldername)
    query_image_features = getQueryImageRep(V, imagePath, modelType)
    list = comparisonHelper.getMSimilarImages(U, query_image_features, m, imagePaths)
    plotFigures(list)


def task5(foldername, folderPath, imagePath):
    print(" EXECUTING TASK 5 ")
    # print(folderPath)
    # print(imagePath)
    initTask5(folderPath, imagePath)


def task6(subjectid, handInfoPath, folderPath):
    print("Executing Task 6")
    helper.getSubjectImages(subjectid)


def task8(imageDir, handInfoCSV, k):
    initTask8(imageDir, handInfoCSV, k)
    print(" EXECUTING TASK 8 ")