import math
import matplotlib.pyplot as plt

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
