import numpy as np
import sys
from sklearn.decomposition import PCA
from skimage.io import imread_collection, imread, imshow, show, imshow_collection
import matplotlib.pyplot as plt
import glob
from random import shuffle
from src.dimReduction.dimRedHelper import DimRedHelper
import pickle
from random import seed
from random import randrange
from csv import reader
import random
import pandas as pd
import os
# from src.common.imageHelper import ImageHelper
from src.constants import BLOCK_SIZE
from src.models.ColorMoments import ColorMoments
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import pickle


# from preprocess import getdata


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left'], row[node['index']]
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right'], row[node['index']]

        # Classification and Regression Tree Algorithm


def decision_tree(train, test, max_depth, min_size):
    print("In decision tree")
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


def f(alpha, labels, kernel, train_data, row, b):
    n = len(labels)
    value = 0.0
    for i in range(n):
        value += alpha[i] * labels[i] * kernel(train_data[i], row)
    value += b
    return value


def getImages():
    li = []
    with open('store/ls_image.pkl', 'rb') as f2:
        li = pickle.load(f2)
    return li


def helper(d):
    liImages = []
    labels = []
    t = 10

    try:
        # with open('src/store/lsh_candidate_images.pkl', 'rb') as f2:
        #     temp = pickle.load(f2)
        #     t = len(temp)
        with open('store/task6_dt_iteration_images.pkl', 'rb') as f2:
            liImages = pickle.load(f2)
        with open('store/task6_dt_iteration_labels.pkl', 'rb') as f2:
            labels = pickle.load(f2)
    except:
        pass

    for img in d['relevant']:
        if img not in liImages:
            liImages.append(img)
            labels.append(1)
        else:
            indx = liImages.index(img)
            labels[indx] = 1

    for img in d['nonrelevant']:
        if img not in liImages:
            liImages.append(img)
            labels.append(0)
        else:
            indx = liImages.index(img)
            labels[indx] = 0

    print("Labels len {} images len {}".format(len(liImages), len(labels)))

    with open('store/task6_dt_iteration_images.pkl', 'wb') as f2:
        pickle.dump(liImages, f2)

    with open('store/task6_dt_iteration_labels.pkl', 'wb') as f2:
        pickle.dump(labels, f2)

    names = getImages()
    obj = DimRedHelper()

    features1 = obj.getDataMatrixForHOG(names, [])
    features2 = obj.getDataMatrixForHOG(liImages, [])

    relevance_labels = []
    train_data = []
    pca = PCA()
    pca.fit(features1)
    ratio = 0.0
    m = 0
    for i in range(len(pca.explained_variance_ratio_)):
        ratio += pca.explained_variance_ratio_[0]
        if ratio >= 0.95:
            m = i
            break
    if m > 10:
        m = 10
    pca = PCA(m)
    features1 = pca.fit_transform(features1)
    features2 = pca.transform(features2)

    for i in range(len(liImages)):
        if labels[i] == 0:
            relevance_labels.append(0)
            train_data.append(features2[i])
            continue
        if labels[i] == 1:
            relevance_labels.append(1)
            train_data.append(features2[i])
    train_data = np.asarray(train_data)

    sample_size = train_data.shape[0]
    node = build_tree(train_data, 10, 1)
    # alpha, b = SMO(C=100.0, tol=1e-4, train_data=train_data, labels=relevance_labels,
    # kernel=Linear_kernel, max_passes=3)

    scores_r = []
    scores_i = []
    scores = []
    for i in range(features1.shape[0]):
        sample = features1[i]
        val, score = predict(node, sample)
        scores.append(score)
        # if(val==1):
        #     scores_r.append(score)
        # else:
        #     scores_i.append(score)
    score_in = list(np.argsort(scores_i))
    score_rn = list(np.argsort(scores_r))
    score_reordered = list(np.argsort(scores))
    # for i in score_in:
    #     score_rn.append(i)

    # score_indexes.reverse()  # decreasing order
    reordered_img_names = [names[index] for index in score_reordered]
    return reordered_img_names[:t]

# if __name__ == '__main__':
#     iteration = 0
#     names = []
#     d = {'relevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006333.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006332.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000005.jpg'], 'nonrelevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006331.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000002.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000003.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000008.jpg']}
#     helper(d)
