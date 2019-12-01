import pandas as pd
from src.common.imageIndex import ImageIndex
from src.distance.distanceArchiver import DistanceArchiver
from src.models.featureArchiver import FeatureArchiver
from src.partition.graphArchiver import GraphArchiver, GraphType
from src.tasks.task3 import Task3
from src.models.enums.models import ModelType
from src.tasks.task4ppr import Task4PPR
from src.tasks.task4pprNoCache import Task4PPRNoCache
from src.constants import CLASSIFICATION_IMG_DIR, CLASSIFICATION_META_CSV
import glob

def init():
    # ImageIndex()
    # imageDir = "/Users/yvtheja/Documents/Dataset2"
    # imagePaths = [
    #     "/Users/yvtheja/Documents/Dataset2/Hand_0006321.jpg",
    #     "/Users/yvtheja/Documents/Dataset2/Hand_0006322.jpg",
    #     "/Users/yvtheja/Documents/Dataset2/Hand_0006323.jpg"
    # ]
    # Task3(10, imageDir, modelTypes=[ModelType.HOG],).visualiseSimilarImages(10, imagePaths)


    # Run below code for pprClassification
    for imagePath in glob.glob("/Users/yvtheja/Documents/Test_Set/*.jpg"):
        print("Image class : {}".format(Task4PPRNoCache(CLASSIFICATION_IMG_DIR, CLASSIFICATION_META_CSV, modelTypes=[ModelType.CM]).getClassForImage(imagePath)))


if __name__ == "__main__":
    init()