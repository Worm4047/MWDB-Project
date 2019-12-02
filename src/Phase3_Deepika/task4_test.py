# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:09:11 2019

@author: Deepika
"""

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
import pandas as pd
import os
import csv
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from skimage import feature
import cv2
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.dimReduction.dimRedHelper import DimRedHelper
from sklearn.model_selection import GridSearchCV


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




def linear(x1, x2, gamma):
    return np.dot(x1, x2)


def poly(x, y, gamma, p=3):
    return (1 + (np.dot(x, y) * gamma)) ** p


def rbf(x, y, gamma):
    return np.exp(-gamma * (linalg.norm(x - y) ** 2))


class SVM(object):

    def __init__(self, kernel=linear, C=None, gamma=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #print(n_samples, n_features)
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                if self.kernel == 'linear':
                    # print(i,j)
                    # print(X[i], X[j])
                    K[i, j] = linear(x_i, x_j, self.gamma)

                    # print("Inside linear")
                elif self.kernel == 'rbf':
                    K[i, j] = rbf(x_i, x_j, self.gamma)
                else:
                    K[i, j] = poly(x_i, x_j, self.gamma)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
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
        #print("a:", a)
        #print(a.shape)
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # print("self.sv_y", self.sv_y)
        # print(self.sv_y.shape)
        # print("sv", self.sv)
        # print(self.sv.shape)
        # print("{} support vectors out of {} points".format(len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            # self.b += self.sv_y.iloc[n]
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)
        #print("b value", self.b)

        # # testing
        # print("self.a[n] rows:", np.size(self.a, 0))
        # # print("self.a[n] cols:",np.size(self.a,1))
        # print("sv_y.iloc[n] rows:", np.size(self.sv_y, 0))
        # # print("sv_y.iloc[n] cols:",np.size(self.sv_y,1))
        # print("self.sv[n] rows:", np.size(self.sv, 0))
        # print("self.sv[n] cols:", np.size(self.sv, 1))

        # print("self.a[0]:", self.a[0])
        # print("sv_y.iloc[0]", self.sv_y[0])
        # print("self.sv[0]", self.sv[0, :])
        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            # print("self.w rows:", np.size(self.w, 0))
            # print("self.w cols:",np.size(self.w,1))
            for n in range(len(self.a)):
                """mat_=self.sv_y.iloc[n]* self.sv.iloc[n,:]
                print("mat_.rows:",np.size(mat_,0))
                #print("mat_.cols:",np.size(mat_,1))
                self.w += self.a[n]* mat_"""
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            # return np.dot(X, self.w) + self.b
            return np.dot(X, self.w) - self.C
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    if self.kernel == 'linear':
                        s += a * sv_y * linear(X[i], sv, self.gamma)
                    elif self.kernel == 'rbf':
                        s += a * sv_y * rbf(X[i], sv, self.gamma)
                    else:
                        s += a * sv_y * poly(X[i], sv, self.gamma)
                y_predict[i] = s
            # return y_predict + self.b
            return y_predict - self.C

    def predict(self, X):
        return np.sign(self.project(X))


if __name__ == "__main__":

    path_labelled_images = input("Enter path to folder with labelled images:")
    path_labelled_metadata = input("Enter path to metadata with labelled images:")
    path_unlabelled_images = input("Enter path to folder with unlabelled images:")
    path_unlabelled_metadata = input("Enter path to metadata with unlabelled images:")
    path_original_metadata = input("Enter path to metadata with original data:")
    path_storing_files = input("Enter path to store feature files:")

    """X_train is the data matrix"""

    imagePaths = glob.glob(os.path.join(path_labelled_images, "*.{}".format('jpg')))
    obj = DimRedHelper()
    X_train = obj.getDataMatrixForLBP(imagePaths, [])

    #print(type(X_train))
    ss = StandardScaler()
    #ss = MinMaxScaler(feature_range=(0, 1))
    X_train = ss.fit_transform(X_train)
    # X_train=pd.DataFrame(X_train)
    pca = PCA(n_components=100)
    X_train = (pca.fit_transform(np.mat(X_train)))


    """Storing the name of images from labelled folder in a csv file as sequence is not same in folder and metadata"""
    df_labelled_images_name = store_image_name(path_labelled_images, "labelled")
    # print(df_labelled_images_name.head())
    df_unlabelled_images_name = store_image_name(path_unlabelled_images, "unlabelled")

    # print(df_labelled_images_name.head())
    # print(df_unlabelled_images_name.head())

    """df_labelled_dataInfo datafram after dividing the aspect of hand column"""
    df_labelled_dataInfo = making_two_columns(path_labelled_metadata)
    # print(df_labelled_dataInfo.head())
    df_labelled_dataInfo = df_labelled_dataInfo[['imageName', 'SideOfHand']].copy()
    # print(df_labelled_dataInfo.head())
    """Merging both dataframes.Important because metadata and folder have different sequence of images"""

    df_labelled_Info = pd.merge(df_labelled_images_name, df_labelled_dataInfo, on="imageName")
    # print("After merging:",df_labelled_Info.tail(30))
    df_labelled_Info["SideOfHand"] = coding(df_labelled_Info["SideOfHand"], {'dorsal': 1, 'palmar': -1})
    """y_training"""
    y_training = df_labelled_Info["SideOfHand"]
    y_training = np.array(y_training)
    #print(type(y_training))
    # print(y_training.head())

    """X_test is the data matrix of unlabelled images"""

    imagePaths = glob.glob(os.path.join(path_unlabelled_images, "*.{}".format('jpg')))
    X_test = obj.getDataMatrixForLBP(imagePaths, [])
    #ss = MinMaxScaler(feature_range=(0, 1))
    #X_test = ss.fit_transform(X_test)
    X_test = ss.transform(X_test)
    # X_test=pd.DataFrame(X_test)
    X_test = (pca.transform(np.mat(X_test)))


    """df_original_dataInfo datafram after dividing the aspect of hand column"""

    df_original_dataInfo = making_two_columns(path_original_metadata)
    # print(df_original_dataInfo.head())
    df_original_dataInfo = df_original_dataInfo[['imageName', 'SideOfHand']].copy()
    # print(df_original_dataInfo.head())
    """Merging both dataframes.Important because metadata and folder have different sequence of images"""

    df_unlabelled_Info = pd.merge(df_unlabelled_images_name, df_original_dataInfo, on="imageName")
    # print("After merging:",df_unlabelled_Info.tail(30))
    df_unlabelled_Info["SideOfHand"] = coding(df_unlabelled_Info["SideOfHand"], {'dorsal': 1, 'palmar': -1})
    """y_test_actual"""
    y_test_actual = df_unlabelled_Info["SideOfHand"]
    # print(y_test_actual.tail(10))

    """y_test_predict will have predicted result"""
    """Training for SVM"""
    model = SVC()
    variance = X_train.var()
    n_features = X_train.shape[1]
    gamma = 1 / (variance * n_features)

    param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.8, 1],
                  'gamma': [gamma],
                  'kernel': ['linear', 'poly', 'rbf']
                  }
    gsearch = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='f1_micro',
                           cv=5,
                           verbose=1000)

    gsearch.fit(X_train, y_training)
    print(gsearch.best_params_)
    #  ------------------------------------------------------------------------------------
    #  finer grid search
    #  ------------------------------------------------------------------------------------
    C = gsearch.best_params_['C']
    #print("C value:", C)
    gamma = gsearch.best_params_['gamma']
    #print("gamma value:", gamma)
    kernel = gsearch.best_params_['kernel']
    #print("kernel value:", kernel)

    param_grid_finer = {'C': np.linspace(C / 2, C * 2, num=10),
                        'gamma': [gamma],
                        'kernel': [kernel]
                        }
    gsearch_finer = GridSearchCV(estimator=model,
                                 param_grid=param_grid_finer,
                                 scoring='f1_micro',
                                 cv=5,
                                 verbose=1000)

    gsearch_finer.fit(X_train, y_training)
    print("After finer grid search:", gsearch_finer.best_params_)


    svc_clf = SVM(**gsearch_finer.best_params_)
    svc_clf.fit(X_train, y_training)
    y_predict = svc_clf.predict(X_test)
    correct = np.sum(np.array(y_predict) == np.array(y_test_actual))
    print("{} out of {} predictions correct".format(correct, len(y_predict)))
    cal_accuracy(y_test_actual, y_predict)
