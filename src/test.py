import numpy as np

if __name__ == "__main__":
    # tempArray = np.zeros((16, 16, 3))
    # np.save("imageName", tempArray)

    array = np.load("imageName.npy")
    print(array.shape)