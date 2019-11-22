
##########################################################################################################
# DistanceArchiver
##########################################################################################################


# for queryIndex, queryImagePath in enumerate(imagePaths):
#     queryStart = time.time()
#     if os.path.exists(self.getDistancesFilenameForImage(queryImagePath, modelType, distanceType)): continue
#
#     queryFeatureVector = self.imageHelper.getImageFeatures(queryImagePath, modelType, retriveshape=True).reshape(1, -1)
#     distances = spatial.distance.cdist(dataMatrix, queryFeatureVector, metric='euclidean')
#     distancesDict["distance"] = distances.tolist()
#
#     # for imageIndex, imagePath in enumerate(imagePaths):
#     #     start = time.time()
#     #     imageFeatureVector = self.featureArchiver.getFeaturesForImage(imagePath, modelType)
#     #     # print("Pair wise distance calculation | Query image number: {} - Image number: {} | Feature calculation time: {}"
#     #     #       .format(queryIndex, imageIndex, time.time() - start))
#     #     distance = self.distanceCalculator.getDistance(queryFeatureVector, imageFeatureVector, distanceType)
#     #
#     #     imageName = self.imageHelper.getImageName(imagePath)
#     #     distancesDict["imageName"].append(imageName)
#     #     distancesDict["imageId"].append(self.imageIndex.getImageId(imageName))
#     #     distancesDict["distance"].append(distance)
#
#     print("Pair wise distance calculation | Processed {} out of {} | Last image time: {}"
#           .format(queryIndex, imagesCount, time.time() - queryStart))
#     distanceDf = pd.DataFrame.from_dict(distancesDict)
#
#     self.savePairWiseDistancesForImage(queryImagePath, distanceDf, modelType, distanceType)

# def getDistancesFilenameForImage(self, imagePath, modelType, distanceType):
#     return os.path.join(DISTANCES_PAIR_WISE_STORE, "{}_{}_{}.csv".format(self.imageHelper.getImageName(imagePath), modelType.name, distanceType.name))

# def savePairWiseDistancesForImage(self, imagePath, distanceDf, modelType, distanceType):
#     distanceDf.to_csv(self.getDistancesFilenameForImage(imagePath, modelType, distanceType))

# def getDistancesForImage(self, imagePath, modelType=ModelType.CM, distanceType=DistanceType.EUCLIDEAN):
#     imageDistanceFilePath = self.getDistancesFilenameForImage(imagePath, modelType, distanceType)
#     if os.path.exists(imageDistanceFilePath):
#         return pd.read_csv(imageDistanceFilePath)
#     else:
#         raise ValueError("No distances file exists for image: {}".format(imagePath))

