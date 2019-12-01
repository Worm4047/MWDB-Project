from src.constants import DORSAL_DATABASE_PATH, PALMAR_DATABASE_PATH
from src.classifiers.pprClassifier import ImageClass
from src.common.imageIndex import ImageIndex
from src.partition.graphArchiver import GraphArchiver
from src.models.enums.models import ModelType
from src.common.imageHelper import ImageHelper
import pandas as pd
import os
import shutil
import glob


class Task4PPR:
    dorsalImagePaths = []
    palmarImagePaths = []
    dorsalGraph = None
    palmarGraph = None
    imageIndexDorsal = None
    imageIndexPalmar = None
    imageHelper = None

    def __init__(self, imageDir, metaDataCSV, modelTypes=None):
        if modelTypes is None:
            modelTypes = [ModelType.HOG]

        self.imageHelper = ImageHelper()
        self.modelTypes = modelTypes
        self.dorsalImagePaths = self.getLabelledDorsalImages(metaDataCSV, imageDir)
        self.palmarImagePaths = self.getLabelledPalmarImages(metaDataCSV, imageDir)
        self.emptyFolder(DORSAL_DATABASE_PATH)
        self.emptyFolder(PALMAR_DATABASE_PATH)
        self.copyImagesToFolder(DORSAL_DATABASE_PATH, self.dorsalImagePaths, imageClass=ImageClass.DORSAL)
        self.copyImagesToFolder(DORSAL_DATABASE_PATH, self.palmarImagePaths, imageClass=ImageClass.PALMAR)
        self.copyImagesToFolder(PALMAR_DATABASE_PATH, self.dorsalImagePaths, imageClass=ImageClass.DORSAL)
        self.copyImagesToFolder(PALMAR_DATABASE_PATH, self.palmarImagePaths, imageClass=ImageClass.PALMAR)


    def emptyFolder(self, folderPath):
        if len(glob.glob(os.path.join(folderPath, "*.jpg"))) > 0:
            shutil.rmtree(folderPath)

    def copyImagesToFolder(self, folderPath, imagePaths, imageClass=ImageClass.PALMAR):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        for imagePath in imagePaths:
            shutil.copy(imagePath, os.path.join(folderPath, "{}_{}.jpg".format(self.imageHelper.getImageName(imagePath), imageClass.name)))

    def removeImageFromFolder(self, folderPath, queryImagePath):
        imagePaths = glob.glob(os.path.join(folderPath, "*.jpg"))

        for imagePath in imagePaths:
            if self.imageHelper.getImageName(imagePath) == self.imageHelper.getImageName(queryImagePath):
                os.remove(imagePath)

    def getLabelledDorsalImages(self, csvpath, imagePath):
        return self.getLabelledImages(csvpath, imagePath, True)

    def getLabelledPalmarImages(self, csvpath, imagePath):
        return self.getLabelledImages(csvpath, imagePath, False)

    def getLabelledImages(self, csvPath, imagePath, dorsal):
        label_df = pd.read_csv(csvPath)
        if dorsal:
            label_df = label_df.loc[label_df['aspectOfHand'].str.contains('dorsal')]
        else:
            label_df = label_df.loc[label_df['aspectOfHand'].str.contains('palmar')]
        images = list(label_df['imageName'].values)
        for i in range(len(images)):
            images[i] = imagePath + "/" + images[i]
        return images

    def getClassForImage(self, queryImagePath):
        dorsalQueryPath = os.path.join(DORSAL_DATABASE_PATH, "query.jpg")
        shutil.copy(queryImagePath, dorsalQueryPath)
        palmarQueryPath = os.path.join(PALMAR_DATABASE_PATH, "query.jpg")
        shutil.copy(queryImagePath, palmarQueryPath)

        self.imageIndexDorsal = ImageIndex(ImageClass.DORSAL)
        self.imageIndexPalmar = ImageIndex(ImageClass.PALMAR)

        self.dorsalGraph = GraphArchiver(28, modelTypes=self.modelTypes, imageClass=ImageClass.DORSAL)
        dorsalPPRs = self.dorsalGraph.getPersonalisedPageRankForThreeImages([self.imageIndexDorsal.getImageIdForPath(imagePath) for imagePath in glob.glob(os.path.join(DORSAL_DATABASE_PATH, "*DORSAL.jpg"))])
        dorsalGraphPR = dorsalPPRs[self.imageIndexDorsal.getImageIdForPath(
            dorsalQueryPath)] / self.dorsalGraph.getImagesCount()

        self.palmarGraph = GraphArchiver(28, modelTypes=self.modelTypes, imageClass=ImageClass.PALMAR)
        palmarPPRs = self.palmarGraph.getPersonalisedPageRankForThreeImages([self.imageIndexPalmar.getImageIdForPath(imagePath) for imagePath in glob.glob(os.path.join(PALMAR_DATABASE_PATH, "*PALMAR.jpg"))])
        palmarGraphPR = palmarPPRs[self.imageIndexDorsal.getImageIdForPath(
            palmarQueryPath)] / self.dorsalGraph.getImagesCount()

        self.dorsalGraph.deleteGraphs()
        self.palmarGraph.deleteGraphs()
        self.imageIndexDorsal.deleteIndexFile()
        self.imageIndexPalmar.deleteIndexFile()

        imageClass = None
        if dorsalGraphPR >= palmarGraphPR:
            imageClass = ImageClass.DORSAL
        else: imageClass = ImageClass.PALMAR

        os.remove(dorsalQueryPath)
        os.remove(palmarQueryPath)

        return imageClass







