import heapq
import numpy as np


# Cosine distance is calculated for 2 values queryImageVector and imageVector
from scipy import spatial


def compareWithCosine(queryImageVector, imageVector):
    # return np.linalg.norm(imageVector - queryImageVector)
    return 1 - spatial.distance.cosine(imageVector, queryImageVector)


# Eucledian distance is calculated for 2 values val1 and val2
def computeWithEucledian(val1, val2):
    return np.linalg.norm(val1 - val2)


# This function takes as input an image and a value K
# We then iterate through the values in the CSV file and compute the distance of each image vector from the input image
# We are using a minHeap to store the values after comparing.
# If the value of the min heap is more than k, we are removing the last element from the min heap
# This function then returns the final K vectors and the
def getMSimilarImages(dataMatrix, query_image_features, m, imageNames):
    h = []
    for row in range(0, dataMatrix.shape[0]-1):
        image_name = imageNames[row]
        image_vector = dataMatrix[row]
        # falttenedValue = [float(x) for x in s[1: -1].split(',')]
        # we are taking the negative value of the distance as there is no min-heap in python
        distance = -computeWithEucledian(image_vector, query_image_features)
        heapq.heappush(h, (distance, image_name))
        print(len(h), m)
        if len(h) > m:
            heapq.heappop(h)
    res = heapq.nlargest(m, h)
    finalRes = {}
    for item in res:
        finalRes[item[1]] = -item[0]
    return finalRes