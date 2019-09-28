import os
import matplotlib.pyplot as plt
import numpy as np
import math

def getImageName(imagePath):
    return os.path.basename(imagePath)

def getCubeRoot(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)

def cleanDirectory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def plotFigures(figures, ncols=5):
    imagesCount = len(figures)
    nrows = math.ceil(imagesCount / ncols)

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    fig.canvas.set_window_title('Similar images')
    for index in range(nrows * ncols):
        axeslist.ravel()[index].set_axis_off()

    for ind, title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)

    plt.tight_layout()
    plt.show()
