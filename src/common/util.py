import numpy as np
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

def sort_print_n_return(npArray):
    ind = np.argsort(npArray, axis=1)
    rows = ind.shape[0]
    cols = ind.shape[1]
    twpair = np.zeros((cols,), dtype = 'i,i')
    for x in range(0, rows):
        l = []
        for y in range(0, cols):
            l.append((ind[x, y], npArray[x,ind[x,y]]))
        twpair = np.vstack([twpair,  np.array(l, dtype='i,i')])
    twpair = np.delete(twpair, (0), axis=0)
    return twpair

def visualize_ec(twpair, type, orgDataMatrix):
    rows = twpair.shape[0]
    cols = twpair.shape[1]
    fig, axes = plt.subplots(rows, cols)
    if type == 'data':
        for x in range(0, rows):
            for y in range(0, cols):
                imageid, imagepath = "image"#get_img_path_from_index(twpair[x, y][0])
                image = Image.open(imagepath)
                axes[x, y].imshow(image)
                axes[x, y].set_title("score:" + twpair[x, y][1], fontsize=7)
        plt.show()
    else:
        fig, axes = plt.subplots(rows, 1)
        projection_component = np.matmul(orgDataMatrix, get_projection_axis(twpair))
        ind = np.argsort(projection_component, axis=1)
        row_pc = ind.shape[0]
        for x in range(0, rows):
            imageid, imagepath = "image"#get_img_path_from_index(ind[x, 0])
            image = Image.open(imagepath)
            axes[x, 0].imshow(image)
        plt.show()

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

def findNMF(vspace, k, initialisation, tol, max_iter):
    model = NMF(n_components=k, init=initialisation, tol = tol, max_iter=max_iter)
    W = model.fit_transform(vspace)
    H = model.components_
    return W, H

