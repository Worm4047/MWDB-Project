from src.dimReduction.interfaces.ReductionModel import ReductionModel

class SVD(ReductionModel):
    def __init__(self, dataMatrix):
        super(SVD, self).__init__(dataMatrix)

    def getDecomposition(self):
        #Implement the function here

        print("Boom")