import numpy as np


class CsvOperation:
    def __init__(self, model, label, decomposition, type):
        self.model = model
        self.label = label
        self.decomposition = decomposition
        self.type = type

    def load_csv(self):
        return np.genfromtxt(self.get_csv_path(), delimiter=',')

    def get_csv_path(self):
        return self.decomposition + "/" + self.model + "_" + self.label + "_" + self.type + ".csv"

    def save_as_csv(self, npArray):
        fname = self.get_csv_path()
        np.savetxt(fname, npArray, delimiter=",")


# if __name__ == "__main__":
#      c = CsvOperation(model="LBP", label="right", decomposition="svd", type = "feature")
#      x = np.asarray([ [1.1,2.2,3.3], [4.4,5.4,6.4], [7.4,8.4,9.4] ])
#      c.save_as_csv(npArray=x)
#      c.load_csv()
