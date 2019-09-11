import os
import cv2
import glob
import datetime
import numpy as np
from models import ColorMoments
from comparators import CMComparator
import csv

if __name__ == "__main__":
    databasePath = "/Users/yvtheja/Documents/Hands"

    dbImagePaths = glob.glob(os.path.join(databasePath, "*.jpg"))
    with open('featureVectorsStore/colorMomentsFeatures.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for index, dbImagePath in enumerate(dbImagePaths):
            print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
            dbImg = cv2.imread(dbImagePath, cv2.IMREAD_COLOR)
            dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
            dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)

            featureVector = np.concatenate((dbImageColorMomments.meanFeatureVector,
                                          dbImageColorMomments.varianceFeatureVector,
                                          dbImageColorMomments.skewFeatureVector), axis=2)

            spamwriter.writerow([dbImagePath, ",".join(featureVector.flatten().astype(np.str)),
                                 ",".join(np.array(featureVector.shape).astype(np.str))])
