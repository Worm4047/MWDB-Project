
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
import glob

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
            img_id, img_vec = images[j], vec
            j+=1
            for idx, hash_vec in enumerate(hash_table):
                buckets = self.hash_obj.hash(img_vec, self.vec[idx], self.b[idx], self.w)
                for i in range(len(buckets)):
                    hash_vec[i][buckets[i]].add(img_id)
        # TODO save hashtable somewhere
        if verbose:
            pprint(hash_table)
        return hash_table

    def init_hash_table(self):
        hash_table = []
        for i in range(self.num_layers):
            hash_layer = []
            for j in range(self.num_hash):
                hash_vec = defaultdict(set)
                hash_layer.append(hash_vec)
            hash_table.append(hash_layer)
        return hash_table

    def find_ann(self, query_point, hash_table, k=5):
        candidate_imgs = set()
        num_conjunctions = self.num_hash
        for layer_idx, layer in enumerate(self.vec):
            hash_vec = hash_table[layer_idx]
            buckets = self.hash_obj.hash(query_point, layer, self.b[layer_idx], self.w)
            cand = hash_vec[0][buckets[0]].copy()
            # print(cand, type(cand))
            # self.test(hash_vec[1])
            for ix, idx in enumerate(buckets[1:num_conjunctions]):
                # needs ix+1 since we already took care of index 0
                cand = cand.intersection(hash_vec[ix + 1][idx])
            candidate_imgs = candidate_imgs.union(cand)
            if len(candidate_imgs) > 4 * k:
                print(f'Early stopping at layer {layer_idx} found {len(candidate_imgs) }')
                break
        if len(candidate_imgs) < k:
            if self.num_hash > 1:
                self.num_hash -= 1
                print('Reduced number of hashes')
                return self.find_ann(query_point, hash_table, k=k)
            else:
                print('fubar')
        return candidate_imgs
    # def init_hash_table(self):
    #     hash_table = []
    #     for i in range(self.num_layers):
    #         hash_layer = []
    #         for j in range(self.num_hash):
    #             # hash_vec = defaultdict(set)
    #             hash_vec = {}
    #             hash_layer.append(hash_vec)
    #         hash_table.append(hash_layer)
    #     return hash_table

    # def find_ann(self, query_point, hash_table, k=5):
    #     # print(candidate_imgs)
    #     candidate_imgs = set()
    #     num_conjunctions = self.num_hash
    #     for layer_idx, layer in enumerate(self.vec):
    #         hash_vec = hash_table[layer_idx]
    #         print(layer_idx)
    #         buckets = self.hash_obj.hash(query_point, layer, self.b[layer_idx], self.w)
    #         print("Layer {} Buckets {}".format(layer_idx, buckets))
    #         cand = hash_vec[0][buckets[0]].copy()
    #         print("cand {}".format(cand))
    #         for ix, idx in enumerate(buckets[1:self.num_hash]):
    #             print(ix, idx)
    #             cand = set(cand)
    #             cand = cand.intersection(set(hash_vec[ix + 1][idx]))
    #         candidate_imgs = candidate_imgs.union(cand)
    #         if len(candidate_imgs) >  4*k:
    #             print(f'Early stopping at layer {layer_idx} found {len(candidate_imgs) }')
    #             break
    #     print("Len of candidate images " , len(candidate_imgs))
    #     if len(candidate_imgs) < k:
    #         if self.num_hash > 1:
    #             self.num_hash -= 1
    #             # print('Reduced number of hashes')
    #             return self.find_ann(query_point, hash_table, k)
    #         else:
    #             # print('fubar')
    #             pass
    #     return candidate_imgs
    def post_process_filter(self, query_point, candidates, images, k):
        distances = []
        for i in range(len(candidates)):
            img_id = images[i]
            cand = candidates[i]
            dist = self.hash_obj.dist(query_point, cand)
            distances.append({'img_id' : img_id, 'dist' : dist})
        li = sorted(distances, key=lambda x: x['dist'])
        saveImages(li)
        return li[:k]

class l2DistHash:

    def hash(self, point, vec, b, w):
        val = np.dot(vec, point) + b
        val = val * 100
        res = np.floor_divide(val, w)
        return res

    def dist(self, point1, point2):
        v = (point1 - point2)**2
        return math.sqrt(sum(v))

def saveImages(li):
    candidate_ids = []

    for elem in li:
        # print(elem)
        img_id = elem['img_id']
        candidate_ids.append(img_id)
    with open('store/ls_image.pkl', 'wb') as f:
        pickle.dump(candidate_ids, f)

def getImages():
    li = []
    with open('store/ls_image.pkl', 'rb') as f:
        li = pickle.load(f)    
    return li
def getImages(csvPath, imagePath):
    label_df = pd.read_csv(csvPath)
    images = list(label_df['imageName'].values)
    for i in range(len(images)):
        images[i] = imagePath + images[i]
    return images

def saveHashTableData(hashTable, l, w, k, vec, b):
    with open('store/hashTable.pkl', 'wb') as f:
        pickle.dump(hashTable, f)
    with open('store/hashTableParams.txt', 'w+') as outfile2:
        str = "{} {} {}".format(l, w, k)
        outfile2.write(str)
    np.save("store/hashTableVec", vec)
    np.save("store/hashTableB", b)


def readHashTable():
    hashTable = ''
    l, w, k = 0, 0, 0
    vec, b = '',''
    with open('store/hashTable.pkl', 'rb') as f:
        hashTable = pickle.load(f) 
    with open('store/hashTableParams.txt', 'r') as outfile2:
        str = outfile2.readline()
        l,w,k = str.split()
        l,w,k = int(l), int(w), int(k)
    vec = np.load("store/hashTableVec.npy")
    b = np.load("store/hashTableB.npy")
    return hashTable, l, w, k, vec, b

def createAndSaveHashTable(k, l, w, dm, images):
    dim = len(dm[0])
    vec = np.random.rand(l, k, dim)
    b = np.random.randint(low=0, high=w, size=(l, k))
    l2_dist_obj = l2DistHash()
    lsh = LSH(hash_obj=l2_dist_obj, num_layers=l, num_hash=k, vec=vec, b=b, w=w)
    hashTable = lsh.create_hash_table(dm, images)
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
    candidate_ids = list(candidate_ids)
    obj = DimRedHelper()
    candidate_vecs = obj.getDataMatrixForHOG(candidate_ids, []) 
    # print(len(candidate_vecs), len(candidate_ids))
    dist_res = lsh.post_process_filter(query_point=queryDm[0], candidates=candidate_vecs, images = candidate_ids, k=t)
    candidate_ids = []

    for elem in dist_res:
        # print(elem)
        img_id = elem['img_id']
        candidate_ids.append(img_id)
    with open('store/lsh_candidate_images.pkl', 'wb') as f:
        pickle.dump(candidate_ids, f)
    return candidate_ids

def helper():
    # Number of hashes per layer
    k = 50
    # Number of layers
    l = 150
    #Similar Imgaes
    t = 20
    # Datamatrix
    path_labelled_images = 'static/Hands/'
    images = []
    for filename in glob.glob(path_labelled_images + "*.jpg"):
        images.append(filename)
    
    images.sort()
    images = images[:1000]
    # for img in images:
    #     print(img)
    # print(images)
    # csvpath = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/labelled_set1.csv'
    # imagePath = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Labelled/Set1/'
    query_image = ['static/Hands/Hand_0006333.jpg']
    # images = getImages(csvpath, imagePath)[:30]
    obj = DimRedHelper()
    dm = []
    dm = obj.getDataMatrixForHOG(images, [])
    queryDm = obj.getDataMatrixForHOG(query_image, [])
    w = 400
    # print(queryDm)
    candidate_ids = getCandidateImages(k, l, w, dm, images, queryDm, t)
    print(len(candidate_ids))

    return query_image[0], candidate_ids

if __name__ == '__main__':
    helper()