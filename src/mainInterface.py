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
            executeTasks.task3(directoryPath, modelType, k, dimTechnique, label)
        else:
            executeTasks.task1(directoryPath, modelType, k, dimTechnique)

    elif taskType == 2 or taskType == 4 or taskType == 5:
        # folderpath and #imagepath
        #folderpath would be folder path inside data folder
        # each of which will list folder names in the form of
        imagePath = helper.getImagePathFromUser()
        folders = helper.getFolderNames('/home/worm/Desktop/ASU/CSE 515/Phase#2/MWDB/src/data')
        foldername, folderpath = helper.listFolderNames(folders)
        # pass values to task 2 dummy
        if taskType == 2:
            executeTasks.task2(foldername, folderpath, imagePath)
        elif taskType == 4:
            executeTasks.task4(foldername, folderpath, imagePath)
        else:
            executeTasks.task5(foldername, folderpath, imagePath)






