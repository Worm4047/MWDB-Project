import numpy as np
import sys

from sklearn.decomposition import PCA

from skimage.io import imread_collection, imread, imshow, show, imshow_collection
import matplotlib.pyplot as plt
import glob
from random import shuffle
from src.dimReduction.dimRedHelper import DimRedHelper


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

if __name__ == '__main__':
    iteration = 0
    names = []
    while iteration < 2:
        if iteration == 0:
            path_labelled_images = 'D:/studies/multimedia and web databases/project/Hands/Hands/'
            names = []
            for filename in glob.glob(path_labelled_images + "*.jpg"):
                names.append(filename)

            shuffle(names)
            names = names[:5]

        obj = DimRedHelper()
        features = obj.getDataMatrixForHOG(names, [])
        labels = []
        plt.ion()
        for i in range(len(names)):
            img = names[i]
            print(img)
            imshow(img)
            a = input("1:relevant, 0:irrelevant, else:unknown")
            if a == '0':
                labels.append(0)
            elif a == '1':
                labels.append(1)
            else:
                labels.append(-1)
        plt.ioff()
        plt.close()
        relevance_labels = []
        train_data = []
        pca = PCA()
        pca.fit(features)
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
        features = pca.fit_transform(features)

        for i in range(len(names)):
            if labels[i] == 0:
                relevance_labels.append(-1)
                train_data.append(features[i])
                continue
            if labels[i] == 1:
                relevance_labels.append(1)
                train_data.append(features[i])
        train_data = np.asarray(train_data)
        # assume relevance is 1, non-relevance is -1
        sample_size = train_data.shape[0]
        alpha, b = SMO(C=100.0, tol=1e-4, train_data=train_data, labels=relevance_labels,
                       kernel=Linear_kernel, max_passes=3)
        scores = []
        for i in range(features.shape[0]):
            sample = features[i]
            score = f(alpha, relevance_labels, Linear_kernel, train_data, sample, b)
            scores.append(score)
        score_indexes = list(np.argsort(scores))
        score_indexes.reverse()  # decreasing order
        reordered_img_names = [names[index] for index in score_indexes]
        print(reordered_img_names)
        plt.figure(num="Ranked images based on svm FBS", figsize=(25, 15))
        rows = len(reordered_img_names) / 5 + 1
        cols = 5
        names = reordered_img_names
        iteration += 1
        for i in range(len(reordered_img_names)):
            plt.subplot(rows, cols, i + 1)
            plt.title("score=%.3f" % scores[score_indexes[i]])
            plt.imshow(imread(reordered_img_names[i]))
            plt.axis('off')
        plt.show()
