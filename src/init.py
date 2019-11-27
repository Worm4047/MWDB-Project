import pandas as pd
from src.common.imageIndex import ImageIndex
from src.distance.distanceArchiver import DistanceArchiver
from src.models.featureArchiver import FeatureArchiver
from src.partition.graphArchiver import GraphArchiver, GraphType
from src.tasks.task3 import Task3
from src.models.enums.models import ModelType

def init():
    ImageIndex()
    FeatureArchiver(modelType=ModelType.HOG)
    ga = GraphArchiver(10)
    imagePaths = [
        "/Users/yvtheja/Documents/Dataset2/Hand_0000171.jpg",
        "/Users/yvtheja/Documents/Dataset2/Hand_0000173.jpg",
        "/Users/yvtheja/Documents/Dataset2/Hand_0000174.jpg"
    ]
    Task3(10, modelType=ModelType.HOG).visualiseSimilarImages(10, imagePaths)


if __name__ == "__main__":
    init()