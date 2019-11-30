import pandas as pd
from src.common.imageIndex import ImageIndex
from src.distance.distanceArchiver import DistanceArchiver
from src.models.featureArchiver import FeatureArchiver
from src.partition.graphArchiver import GraphArchiver, GraphType
from src.tasks.task3 import Task3
from src.models.enums.models import ModelType
from src.tasks.task4ppr import Task4PPR
from src.constants import CLASSIFICATION_IMG_DIR, CLASSIFICATION_META_CSV
import glob

def init():
    # ImageIndex()
    # imagePaths = [
    #     "/Users/yvtheja/Documents/Dataset2/Hand_0011710.jpg",
    #     "/Users/yvtheja/Documents/Dataset2/Hand_0010144.jpg",
    #     "/Users/yvtheja/Documents/Dataset2/Hand_0008978.jpg"
    # ]
    # Task3(10, modelTypes=[ModelType.HOG]).visualiseSimilarImages(10, imagePaths)

    for imagePath in glob.glob("/Users/yvtheja/Documents/Test_Set/*.jpg"):
        print("Image class : {}".format(Task4PPR(CLASSIFICATION_IMG_DIR, CLASSIFICATION_META_CSV, modelTypes=[ModelType.CM]).getClassForImage(imagePath)))


if __name__ == "__main__":
    init()