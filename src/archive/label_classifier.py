import numpy as np
import pandas as pd
import csv
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def making_two_columns():
    df_hands_info=pd.read_csv("D:/studies/multimedia and web databases/project/HandInfo.csv",delimiter=",") ##We need to give the path to handInfo
    new = df_hands_info["aspectOfHand"].str.split(" ", n=1, expand=True)
    df_hands_info["SideOfHand"]= new[0]
    df_hands_info["WhichHand"]= new[1]
    #df_hands_info.drop(columns =["aspectOfHand"], inplace = True)
    print("df_hands_info info:", df_hands_info.columns)
    return df_hands_info

def label_classifier(model,dimRed,dataMatrix_labels,label,ImageName):
    df_hands_info = making_two_columns() ##Dividing the aspectofHand column of HandInfo into two columns "SideOfHand" and "WhichHand"
    ####Some assumptions that are made
    df_Hand_Features_k_semantics=pd.DataFrame(np.dot(U,s)) ###This dataframe has information about features of images in k latent semantics
    ###U and s are from SVD
    df_Hands_Names_Features_k_semantics = pd.merge(df_Hand_names, df_Hand_Features_k_semantics.iloc[:,0:k], left_index=True, right_index=True)
    ### df_Hand_names contains all the name of images which are in the folder

    if label == "with accessories":
        label = 1
        col_name = "accessories"
    elif label == "without accessories":
        label = 0
        col_name = "accessories"
    else:
        label = label
        if label in ["right", "left"]:
            col_name = "WhichHand"
        elif label in ["dorsal", "palmar"]:
            col_name = "SideOfHand"
        else:
            col_name = "gender"

    df_Hands_labels = df_Hands_Names_Features_k_semantics.merge(df_hands_info, left_on='0_x', right_on='imageName',
                                                                how='inner')
    # print(df_Hands_labels)
    df_Hands_labels_specific = df_Hands_labels[df_Hands_labels[col_name] == label] # selecting only those images and their features which are
    #specific to label
    print("Hands specific to label:", df_Hands_labels_specific.iloc[:, 0:k + 1])

    # OneClassSVM Implementation
    # image_row=df_Hands_labels.index[df_Hands_labels.0_x =='Hand_0009712.jpg']
    image_row = df_Hands_labels.loc[df_Hands_labels['0_x'] == ImageName].index[0]
    print(image_row)

    # Apply standard scaler to output from resnet50
    ss = StandardScaler()
    ss.fit(df_Hands_labels_specific.iloc[:, 1:k + 1])
    X_train = ss.transform(df_Hands_labels_specific.iloc[:, 1:k + 1])
    print("Transformation for training data Done")
    print(df_Hands_labels.iloc[image_row, 0:k + 1])
    X_test = ss.transform([df_Hands_labels.iloc[image_row, 1:k + 1]])
    print("Transformation for test data Done")

    oc_svm_clf = svm.OneClassSVM(gamma=0.01, kernel='rbf', nu=0.1)
    oc_svm_clf.fit(X_train) ## Training the data
    oc_svm_preds = oc_svm_clf.predict(X_test)
    print(oc_svm_preds)
    if oc_svm_preds == 1:
        if label == 1:
            print(f"Class of Image {df_Hands_labels.iloc[image_row, 0]} is with accessories")
        elif label == 0:
            print(f"Class of Image {df_Hands_labels.iloc[image_row, 0]} is without accessories")
        else:
            print(f"Class of Image {df_Hands_labels.iloc[image_row, 0]} is {label}")
    else:
        output = {"right": "left", "left": "right"
            , "male": "female", "female": "male"
            , 1: "without accessories"
            , 0: "with accessories"
            , "dorsal": "palmar", "palmar": "dorsal"}
        print(f"Class of Image {df_Hands_labels.iloc[image_row, 0]} is {output[label]}")

