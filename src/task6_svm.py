import numpy as np
import sys
from sklearn.decomposition import PCA
from skimage.io import imread_collection, imread, imshow, show, imshow_collection
import matplotlib.pyplot as plt
import glob
from random import shuffle
from src.dimReduction.dimRedHelper import DimRedHelper
import pickle

def Linear_kernel(x1, x2):
    inner_product = np.dot(x1, x2.T)
    return inner_product


def RBF_kernel(x1, x2):
    global gamma
    index = -1 * (la.norm((x1 - x2)) ** 2) * gamma
    return math.exp(index)


def f(alpha, labels, kernel, train_data, row, b):
    n = len(labels)
    value = 0.0
    for i in range(n):
        value += alpha[i] * labels[i] * kernel(train_data[i], row)
    value += b
    return value


def SMO(C, tol, train_data, labels, kernel, max_passes):
    size = len(labels)
    b = 0
    times = 0
    alpha = np.zeros(size)
    E = np.zeros(size)
    while times < max_passes:
        num_changed_alphas = 0
        for i in range(size):
            E[i] = f(alpha, labels, kernel, train_data, train_data[i], b) - labels[i]
            if ((labels[i] * E[i] < -tol) and (alpha[i] < C)) or (
                    (labels[i] * E[i] > tol) and (alpha[i] > 0)):
                while True:
                    j = np.random.randint(0, size)
                    if i != j:
                        break
                E[j] = f(alpha, labels, kernel, train_data, train_data[j], b) - labels[j]
                old_i = alpha[i]
                old_j = alpha[j]

                if labels[i] != labels[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L == H:
                    continue

                eta = 2 * kernel(train_data[i], train_data[j]) - kernel(train_data[i],
                                                                        train_data[i]) - kernel(
                    train_data[j], train_data[j])
                if eta >= 0:
                    continue

                alpha[j] = alpha[j] - labels[j] * (E[i] - E[j]) / eta
                # clip
                if alpha[j] > H:
                    alpha[j] = H
                if alpha[j] < L:
                    alpha[j] = L
                if abs(alpha[j] - old_j) < 1e-5:
                    continue

                alpha[i] = alpha[i] + labels[i] * labels[j] * (old_j - alpha[j])
                b1 = b - E[i] - labels[i] * (alpha[i] - old_i) * (kernel(train_data[i], train_data[i])) - \
                     labels[i] * (alpha[j] - old_j) * (kernel(train_data[i], train_data[j]))
                b2 = b - E[i] - labels[i] * (alpha[i] - old_i) * (kernel(train_data[i], train_data[j])) - \
                     labels[i] * (alpha[j] - old_j) * (kernel(train_data[j], train_data[j]))
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1

        if num_changed_alphas == 0:
            times += 1
        else:
            times = 0

    return alpha, b

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
        with open('store/lsh_candidate_images.pkl', 'rb') as f2:
            temp = pickle.load(f2)
            t = len(temp)
        with open('store/task6_svm_iteration_images.pkl', 'rb') as f2:
            liImages = pickle.load(f2) 
        with open('store/task6_svm_iteration_labels.pkl', 'rb') as f2:
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

    with open('store/task6_svm_iteration_images.pkl', 'wb') as f2:
        pickle.dump(liImages, f2)

    with open('store/task6_svm_iteration_labels.pkl', 'wb') as f2:
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
            relevance_labels.append(-1)
            train_data.append(features2[i])
            continue
        if labels[i] == 1:
            relevance_labels.append(1)
            train_data.append(features2[i])
    train_data = np.asarray(train_data)

    sample_size = train_data.shape[0]
    alpha, b = SMO(C=100.0, tol=1e-4, train_data=train_data, labels=relevance_labels,
                    kernel=Linear_kernel, max_passes=3)
    scores = []
    for i in range(features1.shape[0]):
        sample = features1[i]
        score = f(alpha, relevance_labels, Linear_kernel, train_data, sample, b)
        scores.append(score)
    print(scores)
    score_indexes = list(np.argsort(scores))
    score_indexes.reverse()  # decreasing order
    print(score_indexes)
    reordered_img_names = [names[index] for index in score_indexes]
    return reordered_img_names[:t]

if __name__ == '__main__':
    iteration = 0
    names = []
    d = {'relevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006333.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006332.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000005.jpg'], 'nonrelevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006331.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000002.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000003.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000008.jpg']}
    helper(d)
