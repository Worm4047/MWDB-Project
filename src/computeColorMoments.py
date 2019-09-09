import os
import cv2
import glob
import datetime
import numpy as np
from models import ColorMoments
from comparators import CMComparator
import csv

if __name__ == "__main__":
    databasePath = "/Users/yvtheja/Downloads/Hands"

    dbImagePaths = glob.glob(os.path.join(databasePath, "*.jpg"))
    with open('colorMomentsFeatures.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for index, dbImagePath in enumerate(dbImagePaths):
            print("Time: {} | Processing: {}".format(datetime.datetime.now(), index))
            dbImg = cv2.imread(dbImagePath, cv2.IMREAD_COLOR)
            dbImageYUV = cv2.cvtColor(dbImg, cv2.COLOR_BGR2LUV)
            dbImageColorMomments = ColorMoments.ColorMoments(dbImageYUV, 100, 100)

            spamwriter.writerow([dbImagePath,
                                 ",".join(dbImageColorMomments.meanFeatureVector.flatten().astype(np.str)),
                                 ",".join(dbImageColorMomments.varianceFeatureVector.flatten().astype(np.str)),
                                 ",".join(dbImageColorMomments.skewFeatureVector.flatten().astype(np.str))]
                                )