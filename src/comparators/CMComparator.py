import numpy as np
import math

class CMComparator:
    def compare(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Features descriptors can not be of different shapes")

        totalValues = a.shape[0]

        aFlatten = a.flatten()
        bFlatten = b.flatten()

        distance = 0
        for index, x in np.ndenumerate(aFlatten):
            distance += math.sqrt(math.pow(x, 2) - math.pow(bFlatten[index], 2))

        return distance/totalValues
