# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:18:31 2019

@author: Deepika
"""

import os
import pandas as pd
#from task4_SVM import SVM
#from task4_SVM import linear, poly, rbf
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from src.dimReduction.dimRedHelper import DimRedHelper

def getFirstModel():
    pass


def trainedModel(svm_obj, images):
    # based on the images names,getb the features

    relevantImages = images['relevant']
    nonRelevantImages = images['nonrelevant']
    obj = DimRedHelper()
    dmrelevantImages = obj.getDataMatrixForHOG(relevantImages, [])
    y_trainrelevant = [1]*len(relevantImages)
    dmnonRelevantImages = obj.getDataMatrixForHOG(nonRelevantImages, [])
    y_trainnonrelevant = [-1]*len(nonRelevantImages)
    X_train = dmrelevantImages
    X_train.extend(dmnonRelevantImages)
    y_train = y_trainrelevant
    y_train.extend(y_trainnonrelevant)

    print(len(X_train), len(y_train))

    




    # X_train = pd.read_csv(path_storing_files + "table_for_labelled_Hand_features.csv", delimiter=",", header=None)
    # X_train.drop(X_train.columns[len(X_train.columns) - 1], axis=1, inplace=True)
    # y_train =
    # x_test =
    # y_test =
    # ss = MinMaxScaler(feature_range=(0, 1))
    # X_train = ss.fit_transform(X_train)
    # X_test = ss.transform(X_test)
    # model = SVC()

    # y_train = np.array(y_train)

    # variance = X_train.var()
    # print("varaiance:", variance)
    # n_features = X_train.shape[1]
    # print("number of features:", n_features)
    # gamma = 1 / (variance * n_features)
    # print("gamma:", gamma)
    # param_grid = {'C': [0.01, 0.1, 0.5, 0.8, 1, 2],
    #               'gamma': [gamma],
    #               'kernel': ['linear', 'poly', 'rbf']
    #               }
    # gsearch = GridSearchCV(estimator=model,
    #                        param_grid=param_grid,
    #                        scoring='f1_micro',
    #                        cv=5,
    #                        verbose=1000)

    # gsearch.fit(X_train, y_train)
    # print(gsearch.best_params_)

    # svc_clf = SVC(**gsearch.best_params_)
    # svc_clf.fit(X_train, y_train)
    # y_predict = svc_clf.predict(X_test)
    # print("type of y_predict:", type(y_predict))

    # palmarImages, dorsalImages = [], []
    # # print(testImageNames)
    # for i in range(len(y_predict)):
    #     print(i)
    #     key = y_predict[i]
    #     img = testImageNames[i]
    #     print(img, key)
    #     if key == -1:
    #         palmarImages.append(img)
    #     else:
    #         dorsalImages.append(img)

    # return palmarImages, dorsalImages


if __name__ == "__main__":
    d = {'relevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006333.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006332.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006331.jpg'], 'nonrelevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000002.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000003.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000005.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000008.jpg']}
    testImageNames = []
    svm_obj = SVC()
    images_list_feedback = []
    # getFirstModel(testImageNames, svm_obj)
    trainedModel(svm_obj, d)
