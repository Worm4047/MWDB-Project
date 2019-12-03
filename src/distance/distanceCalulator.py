from enum import Enum
import numpy as np
from scipy import spatial
from src.models.HOG import HOG
from src.models.ColorMoments import ColorMoments

class DistanceType(Enum):
    COSINE_SIMILARITY=1
    EUCLIDEAN=2

class DistanceCalculator:
    def getDistance(self, a, b, distanceType):
        if not isinstance(distanceType, DistanceType):
            ValueError("distanceType should be of DistanceType Enum")

        if distanceType == DistanceType.EUCLIDEAN:
            return np.linalg.norm(a - b)
        elif distanceType == DistanceType.COSINE_SIMILARITY:
            return spatial.distance.cosine(a.flatten(), b.flatten())

