import pandas as pd
from src.common.imageIndex import ImageIndex
from src.distance.distanceArchiver import DistanceArchiver
from src.models.featureArchiver import FeatureArchiver
from src.partition.graphArchiver import GraphArchiver, GraphType
from src.tasks.task3 import Task3

def init():
    ImageIndex()
    FeatureArchiver(checkArchive=True)
    DistanceArchiver(checkArchive=True)
    ga = GraphArchiver(10)
    imagePaths = [
        "/Users/yvtheja/Documents/Dataset2/Hand_0000004.jpg",
        "/Users/yvtheja/Documents/Dataset2/Hand_0000005.jpg",
        "/Users/yvtheja/Documents/Dataset2/Hand_0000006.jpg"
    ]
    Task3(10).visualiseSimilarImages(10, imagePaths)


if __name__ == "__main__":
    init()