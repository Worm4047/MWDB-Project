from src.task8 import initTask8
from src.common import helper

def task1(directoryPath, modelType, k, dimRecTechnique):
    print(" EXECUTING TASK 1 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)

def task2(foldername, folderPath, imagePath):
    # call a function to get data from folder name
    print(" EXECUTING TASK 2 ")
    print(folderPath)
    print(imagePath)

def task3(directoryPath, modelType, k, dimRecTechnique, label):
    print(" EXECUTING TASK 3 ")
    print(directoryPath)
    print(modelType)
    print(k)
    print(dimRecTechnique)
    print(label)

def task4(foldername, folderPath, imagePath):
    print(" EXECUTING TASK 4 ")
    print(folderPath)
    print(imagePath)

def task5(foldername, folderPath, imagePath):
    print(" EXECUTING TASK 5 ")
    print(folderPath)
    print(imagePath)

def task6(subjectid):
    print("Executing Task 6")
    helper.getSubjectImages(subjectid)

def task8(imageDir, handInfoCSV, k):
    initTask8(imageDir, handInfoCSV, k)
    print(" EXECUTING TASK 8 ")