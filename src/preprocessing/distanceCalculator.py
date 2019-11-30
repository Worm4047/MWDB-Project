from src.models.enums.models import ModelType

class DistanceCalculator:
    def calculatePairWiseDistances(self, imagePathList, modelType):
        if not isinstance(modelType, ModelType):
            raise ValueError("Model type not supported")

        for queryImagePath in imagePathList:
            
            for imagePath in imagePathList:



