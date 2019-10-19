import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import os

# sort_print_n_return takes as parameter a numpy array, which should be k*n or k*m based on if its a data or feature latent semantic.
# The basic catch is latent vectors should be in rows
# it sorts each row by value and creates a (index, value) tuple for each latent semantic which represent the term-weight pair
# so term weight in each latent semantic(row) represent how much data or feature represents that latent semantic/principal axis
def sort_print_n_return(npArray):
    ind = np.flip(np.argsort(npArray, axis=1), 1)
    rows = ind.shape[0]
    cols = ind.shape[1]
    twpair = np.zeros((cols,), dtype = 'i,f')
    for x in range(0, rows):
        l = []
        for y in range(0, cols):
            l.append((ind[x, y], npArray[x,ind[x,y]]))
        twpair = np.vstack([twpair,  np.array(l, dtype='i,f')])
    #need to delete the initialized all zero row
    twpair = np.delete(twpair, (0), axis=0)
    print(twpair.flatten().tolist())
    return twpair

# visualize ec takes as input the term weight pair matrix and type(data or feature) and orgDataMatrix
# from data perspective its straightforward and you display the image contributions(weights) to each latent space and corresponding image
# from feature perspective you take a dot product of each image data(matrix multiplication),
# in the original dimensions to convert to latent feature space and displays the one with the highest contribution
def visualize_ec(twpair, type, orgDataMatrix, image_dir, all_images):
    fil_all_images = []
    for image in all_images:
        imagepath = os.path.join(image_dir, "{}".format(image))
        if not os.path.exists(imagepath):
            continue

        fil_all_images.append(image)

    all_images = fil_all_images
    rows = twpair.shape[0]
    cols = twpair.shape[1]
    if type == 'data':
        fig, axes = plt.subplots(rows, cols)
        for x in range(0, rows):
            for y in range(0, cols):
                imagepath = os.path.join(image_dir, "{}".format(all_images[twpair[x,y][0]]))
                image = Image.open(imagepath)
                axes[x, y].imshow(image)
                axes[x, y].set_title("score:" + str(twpair[x, y][1]), fontsize=7)
        plt.show()
    else:
        fig, axes = plt.subplots(rows, 1)
        projection_component = np.matmul(orgDataMatrix, get_projection_axis(twpair))
        ind = np.flip(np.argsort(projection_component, axis=0),0)
        for x in range(0, rows):
            imagepath = os.path.join(image_dir, "{}".format(all_images[ind[0, x]]))
            image = Image.open(imagepath)
            axes[x].imshow(image)
        plt.show()

#retrieve the projection axis back from twpair to perform matrix multiplication which is equalent to dot product
def get_projection_axis(twpair):
    rows =  twpair.shape[0]
    columns = twpair.shape[1]
    res = np.zeros(columns,)
    for x in range(0, rows):
        list = twpair[x].tolist()
        list.sort(key=lambda x: x[0])
        matrixColumn = [x[1] for x in list]
        res = np.column_stack((res, np.array(matrixColumn)))
    res = np.delete(res, (0), axis=1)
    return res