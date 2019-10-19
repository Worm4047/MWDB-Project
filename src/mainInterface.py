from src.common import helper
from src import executeTasks


if __name__ == '__main__':
    taskType = int(helper.getTaskFromUser())

    if taskType == 1 or taskType == 3:
        directoryPath = helper.getDatabasePath()
        modelType = helper.getModelFromUser()
        dimTechnique = helper.getDimTechniqueFromUser()
        k = helper.getKFromUser()
        if taskType == 3:
            label = helper.getLabelFromUser()
            handInfoPath = helper.getMetadataPath()
            executeTasks.task3(directoryPath, modelType, k, dimTechnique, label, handInfoPath)
        else:
            executeTasks.task1(directoryPath, modelType, k, dimTechnique)

    elif taskType == 2 or taskType == 4 or taskType == 5:
        # folderpath and #imagepath
        # folderpath would be folder path inside data folder
        # each of which will list folder names in the form of
        imagePath = helper.getImagePathFromUser()
        folders = helper.getFolderNames('store/latentSemantics/')
        foldername, folderpath = helper.listFolderNames(folders)
        # pass values to task 2 dummy
        if taskType == 2 or taskType == 4:
            m = helper.getMFromUser()
            if taskType == 2:
                executeTasks.task2(foldername, folderpath, imagePath, m)
            elif taskType == 4:
                executeTasks.task4(foldername, folderpath, imagePath, m)
        if taskType == 2:
            m = helper.getMFromUser()
            executeTasks.task2(foldername, folderpath, imagePath, m)
        elif taskType == 4:
            m = helper.getMFromUser()
            executeTasks.task4(foldername, folderpath, imagePath, m)
        else:
            executeTasks.task5(foldername, folderpath, imagePath)

    elif taskType == 6:
        subjectId = helper.getSubjectId()
        handInfoPath = helper.getMetadataPath()
        executeTasks.task6(subjectId, handInfoPath)

    elif taskType == 7:
        print("Executing task 7")

    elif taskType == 8:
        imageDir = helper.getDatabasePath()
        handInfoPath = helper.getMetadataPath()
        k = helper.getKFromUser()
        executeTasks.task8(imageDir, handInfoPath, k)







