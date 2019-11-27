# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:09:11 2019

@author: Deepika
"""



from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
import numpy as np
import os
import csv
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pylab as pl
from src.dimReduction.dimRedHelper import DimRedHelper
from src.models.enums.models import ModelType



def making_two_columns(path):
    df_hands_info = pd.read_csv(path, delimiter=",");
    new = df_hands_info["aspectOfHand"].str.split(" ", n=1, expand=True)
    df_hands_info["SideOfHand"] = new[0]
    df_hands_info["WhichHand"] = new[1]
    # df_hands_info.drop(columns =["aspectOfHand"], inplace = True)
    return df_hands_info


def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded


def store_image_name(path, string):
    table_name = "table_" + string + "_Hand_names.csv"
    with open(path + table_name, mode='w+') as table_for_Hand_names:
        table_for_Hand_names_writer = csv.writer(table_for_Hand_names, delimiter=',', quotechar='"',
                                                 quoting=csv.QUOTE_MINIMAL);
        for filename in glob.glob(path + "*.jpg"):
            file_name = os.path.basename(filename);
            # file_name=file_name.replace(".txt",".jpg");
            table_for_Hand_names_writer.writerow([file_name]);

    df_Hand_names = pd.read_csv(path + table_name, delimiter=",", header=None);
    df_Hand_names.columns = ["imageName"]
    return df_Hand_names


def cal_accuracy(Y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(Y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(Y_test, y_pred) * 100)

    print("Report : ",
          classification_report(Y_test, y_pred))

def plot_margin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    # w.x + b = 0
    a0 = -4; a1 = f(a0, clf.w, clf.b)
    b0 = 4; b1 = f(b0, clf.w, clf.b)
    pl.plot([a0,b0], [a1,b1], "k")

    # w.x + b = 1
    a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0,b0], [a1,b1], "k--")

    # w.x + b = -1
    a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0,b0], [a1,b1], "k--")

    pl.axis("tight")
    pl.show()

def plot_contour(X1_train, X2_train, clf):
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    pl.axis("tight")
    pl.show()


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print ("{} support vectors out of {} points".format(len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":

    path_labelled_images = input("Enter path to folder with labelled images:")
    path_labelled_metadata = input("Enter path to metadata with labelled images:")
    path_unlabelled_images = input("Enter path to folder with unlabelled images:")
    path_unlabelled_metadata = input("Enter path to metadata with unlabelled images:")
    path_original_metadata = input("Enter path to metadata with original data:")


    """X_train is the data matrix"""
    obj1=DimRedHelper()
    X_train=(obj1.getDataMatrix(None, ModelType.HOG, label=None, directoryPath = path_labelled_images))
    print("X_train:",X_train)

    """Storing the name of images from labelled folder in a csv file as sequence is not same in folder and metadata"""
    df_labelled_images_name = store_image_name(path_labelled_images, "labelled")
    print(df_labelled_images_name.head())
    df_unlabelled_images_name = store_image_name(path_unlabelled_images, "unlabelled")

    # print(df_labelled_images_name.head())
    print(df_unlabelled_images_name.head())

    """df_labelled_dataInfo dataframe after dividing the aspect of hand column"""
    df_labelled_dataInfo = making_two_columns(path_labelled_metadata)
    print(df_labelled_dataInfo.head())
    df_labelled_dataInfo = df_labelled_dataInfo[['imageName', 'SideOfHand']].copy()
    print(df_labelled_dataInfo.head())
    """Merging both dataframes.Important because metadata and folder have different sequence of images"""

    df_labelled_Info = pd.merge(df_labelled_images_name, df_labelled_dataInfo, on="imageName")
    print("After merging:", df_labelled_Info.tail(30))
    df_labelled_Info["SideOfHand"] = coding(df_labelled_Info["SideOfHand"], {'dorsal': 1, 'palmar': -1})

    """y_training"""
    y_training = df_labelled_Info["SideOfHand"]
    print(y_training.head())

    """df_original_dataInfo dataframe after dividing the aspect of hand column"""

    df_original_dataInfo = making_two_columns(path_original_metadata)
    print(df_original_dataInfo.head())
    df_original_dataInfo = df_original_dataInfo[['imageName', 'SideOfHand']].copy()
    print(df_original_dataInfo.head())

    """Merging both dataframes.Important because metadata and folder have different sequence of images"""

    df_unlabelled_Info = pd.merge(df_unlabelled_images_name, df_original_dataInfo, on="imageName")
    print("After merging:", df_unlabelled_Info.tail(30))
    df_unlabelled_Info["SideOfHand"] = coding(df_unlabelled_Info["SideOfHand"], {'dorsal': 1, 'palmar': -1})
    """y_test_actual"""
    y_test_actual = df_unlabelled_Info["SideOfHand"]
    print(y_test_actual.head())

    """clf = SVM()
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("{} out of {} predictions correct".format(correct, len(y_predict)))

    plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf)"""

    """y_test_predict will have predicted result"""
    """Calculating accuracy"""
    #cal_accuracy(y_test_actual, y_test_predict)



