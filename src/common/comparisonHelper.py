import heapq
import numpy as np


# Cosine distance is calculated for 2 values queryImageVector and imageVector
def compareWithCosine(queryImageVector, imageVector):
    return np.linalg.norm(imageVector - queryImageVector)


# Eucledian distance is calculated for 2 values val1 and val2
def computeWithEucledian(val1, val2):
    sum = 0
    for v1, v2 in zip(val1, val2):
        sum += (v1 - v2) * (v1 - v2)
    return np.math.sqrt(sum)


# This function takes as input an image and a value K
# We then iterate through the values in the CSV file and compute the distance of each image vector from the input image
# We are using a minHeap to store the values after comparing.
# If the value of the min heap is more than k, we are removing the last element from the min heap
# This function then returns the final K vectors and the
def getKSimilarImages(dataMatrix, k, query_image_features, imageNames):
    h = []
    for row in range(0, dataMatrix.shape[0]-1):
        image_name = imageNames[row]
        image_vector = dataMatrix[row]
        # falttenedValue = [float(x) for x in s[1: -1].split(',')]
        # we are taking the negative value of the distance as there is no min-heap in python
        distance = -compareWithCosine(image_vector, query_image_features)
        heapq.heappush(h, (distance, image_name))
        if len(h) > k:
            heapq.heappop(h)
    res = heapq.nlargest(k, h)
    finalRes = []
    for item in res:
        finalRes.append((-item[0], item[1]))
    return finalRes