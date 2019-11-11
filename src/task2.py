import glob
from src.dimReduction.dimRedHelper import DimRedHelper
from src.models.enums.models import ModelType
from sklearn.cluster import KMeans
import numpy as np

# This is a temp function and is used for testing only
def helper():
    path = '/home/worm/Desktop/ASU/CSE 515/MWDB-Project/src/images/'
    images = glob.glob(path + "*.jpg")
    modelType = ModelType.LBP
    obj = DimRedHelper()
    dataMatrix = obj.getDataMatrixForLBP(images, [])
    dataMatrix = np.stack( dataMatrix, axis=0 )
    task2(dataMatrix, images)

# This is the main function
# Input : Datamatrix computed for the images, Image paths (absolute)
# Function :  Performs clustering of the image then visualizes them
def task2(dm, images, c=10):
    print(dm.shape, len(images))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(dm)
    clusters = kmeans.cluster_centers_
    for cluster in clusters:
        print(cluster.shape)

helper()