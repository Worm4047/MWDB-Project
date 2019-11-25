
import os
import json
from matplotlib import pyplot as plt
from collections import defaultdict
from pprint import pprint
import math
import pandas as pd
import numpy as np
from src.dimReduction.dimRedHelper import DimRedHelper
import pickle

class LSH:

    def __init__(self, hash_obj, num_layers, num_hash, vec, b, w):
        self.hash_obj = hash_obj
        self.num_layers = num_layers
        self.num_hash = num_hash
        self.vec = vec
        self.b = b
        self.w = w

    def create_hash_table(self, img_vecs, images, verbose=False):
        """ Vectorized hash function to bucket all img vecs

            Returns
            -------
            hash_table : List of List of defaultdicts


        """
        hash_table = self.init_hash_table()
        j = 0
        for vec in img_vecs:
            img_id, img_vec = images[j], vec[0:]
            j += 1
            for idx, hash_vec in enumerate(hash_table):
                buckets = self.hash_obj.hash(img_vec, self.vec[idx], self.b[idx], self.w)
                for i in range(len(buckets)):
                    if(buckets[i] not in hash_vec[i]):
                        hash_vec[i][buckets[i]] = []
                    if img_id not in hash_vec[i][buckets[i]]:
                        hash_vec[i][buckets[i]].append(img_id)
        # TODO save hashtable somewhere
        # if verbose:
        #     pprint(hash_table)
        return hash_table

    def init_hash_table(self):
        hash_table = []
        for i in range(self.num_layers):
            hash_layer = []
            for j in range(self.num_hash):
                # hash_vec = defaultdict(set)
                hash_vec = {}
                hash_layer.append(hash_vec)
            hash_table.append(hash_layer)
        return hash_table

    def find_ann(self, query_point, hash_table, k=5, candidate_imgs = set()):
        # print(candidate_imgs)
        num_conjunctions = self.num_hash
        for layer_idx, layer in enumerate(self.vec):
            hash_vec = hash_table[layer_idx]
            
            buckets = self.hash_obj.hash(query_point, layer, self.b[layer_idx], self.w)
            # try:
            cand = hash_vec[0][buckets[0]].copy()
            # self.test(hash_vec[1])
            for ix, idx in enumerate(buckets[1:num_conjunctions]):
                # needs ix+1 since we already took care of index 0
                cand = set(cand)
                cand = cand.intersection(set(hash_vec[ix + 1][idx]))
            candidate_imgs = candidate_imgs.union(cand)
            if len(candidate_imgs) >=  k:
                # print(f'Early stopping at layer {layer_idx} found {len(candidate_imgs) }')
                return candidate_imgs
            # except:
            #     pass
        if len(candidate_imgs) < k:
            if num_conjunctions > 1:
                self.num_hash -= 1
                # print('Reduced number of hashes')
                return self.find_ann(query_point, hash_table, k, candidate_imgs)
            else:
                # print('fubar')
                pass
        return candidate_imgs

class l2DistHash:

    def hash(self, point, vec, b, w):
        val = np.dot(vec, point) + b
        val = val * 100
        res = np.floor_divide(val, w)
        return res

    def dist(self, point1, point2):
        v = (point1 - point2)**2
        return math.sqrt(sum(v))

def getImages(csvPath, imagePath):
    label_df = pd.read_csv(csvPath)
    images = list(label_df['imageName'].values)
    for i in range(len(images)):
        images[i] = imagePath + images[i]
    return images

def saveHashTableData(hashTable, l, w, k, vec, b):
    with open('src/store/hashTable.pkl', 'wb') as f:
        pickle.dump(hashTable, f)
    with open('src/store/hashTableParams.txt', 'w+') as outfile2:
        str = "{} {} {}".format(l, w, k)
        outfile2.write(str)
    np.save("src/store/hashTableVec", vec)
    np.save("src/store/hashTableB", b)


def readHashTable():
    hashTable = ''
    l, w, k = 0, 0, 0
    vec, b = '',''
    with open('src/store/hashTable.pkl', 'rb') as f:
        hashTable = pickle.load(f) 
    with open('src/store/hashTableParams.txt', 'r') as outfile2:
        str = outfile2.readline()
        l,w,k = str.split()
        l,w,k = int(l), int(w), int(k)
    vec = np.load("src/store/hashTableVec.npy")
    b = np.load("src/store/hashTableB.npy")
    return hashTable, l, w, k, vec, b

def createAndSaveHashTable(k, l, w, dm, images):
    dim = len(dm[0])
    vec = np.random.rand(l, k, dim)
    b = np.random.randint(low=0, high=w, size=(l, k))
    l2_dist_obj = l2DistHash()
    lsh = LSH(hash_obj=l2_dist_obj, num_layers=l, num_hash=k, vec=vec, b=b, w=w)
    hashTable = lsh.create_hash_table(dm, images, verbose=True)
    saveHashTableData(hashTable, l, w, k, vec, b)
    return hashTable, lsh

def getCandidateImages(k, l, w, dm, images, queryDm, t):
    hashTable = ''
    lsh = ''
    try:
        print("Reading hashtable")
        hashTable, l, w, k, vec, b = readHashTable()
        # pprint(hashTable)
        l2_dist_obj = l2DistHash()
        lsh = LSH(hash_obj=l2_dist_obj, num_layers=l, num_hash=k, vec=vec, b=b, w=w)
    except:
        print("Creating and saving hash table")
        hashTable, lsh = createAndSaveHashTable(k, l, w, dm, images)
    candidate_ids = lsh.find_ann(query_point=queryDm[0], hash_table=hashTable, k=t)
    print(candidate_ids)
    return candidate_ids

def helper():
    # Number of hashes per layer
    k = 30
    # Number of layers
    l = 100
    #Similar IMgaes
    t = 10
    # Datamatrix
    csvpath = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/labelled_set1.csv'
    imagePath = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Labelled/Set1/'
    query_image = ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Labelled/Set1/Hand_0006333.jpg']
    images = getImages(csvpath, imagePath)
    obj = DimRedHelper()
    dm = obj.getDataMatrixForHOG(images, [])
    queryDm = obj.getDataMatrixForHOG(query_image, [])
    w = 400
    candidate_ids = getCandidateImages(k, l, w, dm, images, queryDm, t)
    return query_image[0], candidate_ids
if __name__ == '__main__':
    helper()