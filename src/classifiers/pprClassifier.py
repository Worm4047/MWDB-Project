from enum import Enum

class ImageClass(Enum):
    DORSAL=1
    PALMAR=2
    NONE=3

# class PPRClassifier:
#     graphPos = None
#     graphNeg = None
#     imageIndexNeg = None
#     imageIndexPos = None
#
#     def __init__(self, graphPos, imageIndexPos, graphNeg, imageIndexNeg):
#         self.graphPos = graphPos
#         self.graphNeg = graphNeg
#
#     def getClassForImagePath(self, queryImagePath):
#
#
#         return ImageClass.DORSAL
