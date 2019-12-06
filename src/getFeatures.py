import glob
import os
import csv
import numpy as np
import pandas as pd
import json
import scipy
import matplotlib
import networkx as nx
import seaborn as sns
from src.dimReduction.dimRedHelper import DimRedHelper
from src.models.enums.models import ModelType
from src.common import comparisonHelper
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from scipy import linalg as LA
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import pickle
from src.task5 import helper

def getImageName(img_path):
    imagename = os.path.basename(img_path)
    return imagename


def saveFeatures():
    path_labelled_images = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/'
    images = []
    for filename in glob.glob(path_labelled_images + "*.jpg"):
        images.append(filename)
    # images = images[:10]
    obj = DimRedHelper()
    i = 0
    for img in images:
        print("Processing {} image".format(i+1))
        i += 1
        imgli = [img]
        dmi = obj.getDataMatrixForHOG(imgli, [])
        img_name = getImageName(img)
        li = [img]
        li.extend(dmi)
        linp = np.array(li)
        with open('src/store/features/'+img_name+'.pkl', 'wb') as f:
            pickle.dump(linp, f)
        # res.append(li)
    # resnp = np.array(res)
 

def readFeatures():
    path = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/store/features/'
    features = []
    i = 0
    for filename in glob.glob(path + "*.pkl"):
        if i == 20:
            break
        i += 1
        with open(filename, 'rb') as f:
            resnp = pickle.load(f)
            features.append(resnp)
            # print(resnp[0])
    features = np.array(features)
    # print(features[:,0])
    return features
    # resnp = ''
    # 
    # return resnp 

if __name__ == '__main__':
    readFeatures()
    # saveFeatures2()
    # # Number of hashes per layer
    # k = 50
    # # Number of layers
    # l = 150
    # #Similar Imgaes
    # t = 20
    # res = readFeatures()
    # query_image = ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006333.jpg']
    # obj = DimRedHelper()
    # dm = []
    # # dm = obj.getDataMatrixForHOG(images, [])
    # queryDm = obj.getDataMatrixForHOG(query_image, [])
    # w = 400
    # # print(queryDm)
    # candidate_ids = getCandidateImages(k, l, w, dm, images, queryDm, t)




