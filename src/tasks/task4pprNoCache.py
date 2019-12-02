from src.constants import DORSAL_DATABASE_PATH, PALMAR_DATABASE_PATH
from src.classifiers.pprClassifier import ImageClass
from src.common.imageIndex import ImageIndex
from src.common.imageIndexNoCache import ImageIndexNoCache
from src.partition.graphArchiverNoCache import GraphArchiverNoCache
from src.models.enums.models import ModelType
from src.common.imageHelper import ImageHelper
import pandas as pd
import os
import shutil
import glob


class Task4PPRNoCache:
    dorsalImagePaths = []
    palmarImagePaths = []
    dorsalGraph = None
    palmarGraph = None
    imageIndexDorsal = None
    imageIndexPalmar = None
    imageHelper = None
    imagePaths = None
    imageIndex = None

    def __init__(self, imageDir, metaDataCSV, modelTypes=None):
        if modelTypes is None:
            modelTypes = [ModelType.HOG]

        self.imageHelper = ImageHelper()
        self.modelTypes = modelTypes
        self.dorsalImagePaths = self.getLabelledDorsalImages(metaDataCSV, imageDir)
        dorsalClasses = [ImageClass.DORSAL for _ in self.dorsalImagePaths]
        self.palmarImagePaths = self.getLabelledPalmarImages(metaDataCSV, imageDir)
        palmarClasses = [ImageClass.PALMAR for _ in self.palmarImagePaths]
        self.imagePaths = self.dorsalImagePaths + self.palmarImagePaths
        self.imageClasses = dorsalClasses + palmarClasses

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
        self.imagePaths.append(queryImagePath)
        self.imageClasses.append(ImageClass.NONE)

        imageIndex = ImageIndexNoCache(self.imagePaths, self.imageClasses)
        self.graph = GraphArchiverNoCache(k=len(self.imagePaths), imagePaths=self.imagePaths,
                                                imageClasses=self.imageClasses, modelTypes=self.modelTypes)
        pageRanks = self.graph.getPersonalisedPageRankForImages([imageIndex.getImageIdForPath(queryImagePath)])

        dorsalProb = 0
        for dorsalImagePath in self.dorsalImagePaths:
            dorsalProb += pageRanks[imageIndex.getImageIdForPath(dorsalImagePath)]

        palmarProb = 0
        for palmarImagePath in self.palmarImagePaths:
            palmarProb += pageRanks[imageIndex.getImageIdForPath(palmarImagePath)]

        if dorsalProb >= palmarProb:
            imageClass = ImageClass.DORSAL
        else: imageClass = ImageClass.PALMAR

        self.imagePaths = self.imagePaths[:-1]
        self.imageClasses = self.imageClasses[:-1]

        return imageClass







