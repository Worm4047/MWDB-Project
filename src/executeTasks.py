from src.task8 import initTask8
from src.dimReduction import dimRedHelper
from src.common import dataMatrixHelper
from src.common import util
from src.common import helper
from src.task5 import initTask5_2

#need to test
def task1(directoryPath, modelType, k, dimRecTechnique):
    print(" EXECUTING TASK 1 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    data_matrix = dimRedHelper.getDataMatrix(None, modelType, None, directoryPath)
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
    rpint(m)

#need to test
def task3(directoryPath, modelType, k, dimRecTechnique, label, handInfoPath):
    print(" EXECUTING TASK 3 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    print(label)
    data_matrix = dimRedHelper.getDataMatrix(None, modelType, label, directoryPath)
    images_list_with_label = helper.getImageIdsWithLabelInputs(label, handInfoPath)
    print(images_list_with_label)
    import os
    all_images = []
    for file in os.listdir(directoryPath):
        if file.endswith('.jpg'):
            all_images.append(file)
    print(all_images)
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
    print(m)

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