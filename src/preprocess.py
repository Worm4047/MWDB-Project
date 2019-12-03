import numpy as np
from numpy import linalg
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

from src.common.imageHelper import ImageHelper
from src.constants import BLOCK_SIZE
from src.models.ColorMoments import ColorMoments

from random import seed
from random import randrange
from csv import reader
import random
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt

class HistogramOfGradients:
    
    def __init__(self,numBins,CellSize,BlockSize):
        self.numBins=numBins
        self.CellSize=CellSize
        self.BlockSize=BlockSize
        
    def Change_Image_Size(self,img):
        resized_img=block_reduce(img, block_size=(10, 10, 1), func=np.mean)
        return resized_img
   
    def describe(self,img):
        feature_vector,hog_image=feature.hog(img,orientations=self.numBins,pixels_per_cell=(self.CellSize, self.CellSize), cells_per_block=(self.BlockSize, self.BlockSize),block_norm='L2-Hys',visualize=True,feature_vector=True, multichannel=True)
        return feature_vector,hog_image

def read_images_store_features(path_labelled_images, path_storing_files):
    files = glob.glob(path_storing_files + "*.txt")
    for f in files:
        os.remove(f)
    for filename in glob.glob(path_labelled_images+"*.jpg"):
        #img=cv2.imread(filename,0)##LBP
        img_name=os.path.basename(filename)
        obj = ImageHelper()
        feature_vector = ColorMoments(obj.getYUVImage(filename), BLOCK_SIZE, BLOCK_SIZE).getFeatures()
        
        # img=cv2.imread(filename)##HOG
        # HOG_object=HistogramOfGradients(9,8,2)
        # Resized_image=HOG_object.Change_Image_Size(img)
        # feature_vector,image_HOG=HOG_object.describe(Resized_image)
        
        #np.save("D:/studies/multimedia and web databases/project/feature_vector_"+img_name+".npy", feature_vector)
        np.savetxt(path_storing_files+img_name+".txt", feature_vector, fmt='%f',delimiter=',',newline=',',footer='')
    print("Finished executing read_image")
    return
def read_features_put_dataframe(path_storing_files, string):
    with open(path_storing_files + "table_for_" + string + "_Hand_features.csv", mode='w+') as table_for_Hand_features:
        table_for_Hand_features_writer = csv.writer(table_for_Hand_features, delimiter=',', quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL)
        for filename in glob.glob(path_storing_files + "*.txt"):
            with open(filename, 'r') as in_file:
                stripped = (line.strip() for line in in_file)
                lines = (line.split(",") for line in stripped if line)
                table_for_Hand_features_writer.writerows(lines)
    print("Finished executing read_features")
    return

def store_image_name(path, string):
    table_name = "table_" + string + "_Hand_names.csv"
    with open(path + table_name, mode='w+') as table_for_Hand_names:
        table_for_Hand_names_writer = csv.writer(table_for_Hand_names, delimiter=',', quotechar='"',
                                                 quoting=csv.QUOTE_MINIMAL)
        for filename in glob.glob(path + "*.jpg"):
            file_name = os.path.basename(filename)
            # file_name=file_name.replace(".txt",".jpg");
            table_for_Hand_names_writer.writerow([file_name])
    df_Hand_names = pd.read_csv(path + table_name, delimiter=",", header=None)
    df_Hand_names.columns = ["imageName"]
    return df_Hand_names

def making_two_columns(path):
    df_hands_info = pd.read_csv(path, delimiter=",")
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


def getdata():
    print("In get data")
    path_labelled_images = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Labelled/Set1/'
    path_labelled_metadata = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/labelled_set1.csv'
    path_unlabelled_images = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Unlabelled/Set 1/'
    path_unlabelled_metadata = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/unlabelled_set1.csv'
    path_original_metadata = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/HandInfo.csv'
    path_storing_files = '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/store/imgstore'
    # path_labelled_images = input("Enter path to folder with labelled images:")
    #'phase3_sample_data/Labelled/Set2/'
    # path_labelled_metadata = input("Enter path to metadata with labelled images:")
    #'phase3_sample_data/labelled_set2.csv'
    # path_unlabelled_images = input("Enter path to folder with unlabelled images:")
    #'phase3_sample_data/Unlabelled/Set 2/'
    # path_unlabelled_metadata = input("Enter path to metadata with unlabelled images:")
    #'phase3_sample_data/unlabelled_set2.csv'
    # path_original_metadata = input("Enter path to metadata with original data:")
    #'phase3_sample_data/HandInfo.csv'
    # path_storing_files = input("Enter path to store feature files:")
    #'feature/'

    """X_train is the data matrix"""
    read_images_store_features(path_labelled_images, path_storing_files)
    read_features_put_dataframe(path_storing_files, "labelled")
    X_train = pd.read_csv(path_storing_files + "table_for_labelled_Hand_features.csv", delimiter=",", header=None)
    X_train.drop(X_train.columns[len(X_train.columns) - 1], axis=1, inplace=True)
    pca = PCA(n_components=50)
    X_train = (pca.fit_transform(np.mat(X_train)))

    """Storing the name of images from labelled folder in a csv file as sequence is not same in folder and metadata"""
    df_labelled_images_name = store_image_name(path_labelled_images, "labelled")
    #print("labelled images head: ",df_labelled_images_name.head())
    df_unlabelled_images_name = store_image_name(path_unlabelled_images, "unlabelled")

    # print(df_labelled_images_name.head())
    #print("Unlabelled images head: ",df_unlabelled_images_name.head())

    """df_labelled_dataInfo datafram after dividing the aspect of hand column"""
    df_labelled_dataInfo = making_two_columns(path_labelled_metadata)
    #print("labelled dat info head: ",df_labelled_dataInfo.head())
    df_labelled_dataInfo = df_labelled_dataInfo[['imageName', 'SideOfHand']].copy()
    #print("labelled dat info head2: ",df_labelled_dataInfo.head())
    """Merging both dataframes.Important because metadata and folder have different sequence of images"""

    df_labelled_Info = pd.merge(df_labelled_images_name, df_labelled_dataInfo, on="imageName")
    #print("After merging:",df_labelled_Info.tail(30))
    df_labelled_Info["SideOfHand"] = coding(df_labelled_Info["SideOfHand"], {'dorsal': 1, 'palmar': 0})
    """y_training"""
    y_training = df_labelled_Info["SideOfHand"]
    #print(y_training)
    y_train=y_training.to_numpy()
    #print(y_train)
    ty = np.column_stack((X_train , y_train ))
    #print(ty)
    result_file= open('train_data.csv','w', newline='')
    wr = csv.writer(result_file, dialect='excel')
    for row in ty:
        tem_l = row.tolist()
        wr.writerow(tem_l)
    # print(y_training.head())

    """X_test is the data matrix of unlabelled images"""
    read_images_store_features(path_unlabelled_images, path_storing_files)
    read_features_put_dataframe(path_storing_files, "unlabelled")
    X_test = pd.read_csv(path_storing_files + "table_for_unlabelled_Hand_features.csv", delimiter=",", header=None)
    X_test.drop(X_test.columns[len(X_test.columns) - 1], axis=1, inplace=True)
    # X_test=pd.DataFrame(X_test)
    pca = PCA(n_components=50)
    X_test = (pca.fit_transform(np.mat(X_test)))
    # v = pd.DataFrame(pca.components_)
    # print(X_test.shape)
    # print(v.shape)
    # print(X_test.head())
    result_file1= open('test_data.csv','w', newline='')
    wr1 = csv.writer(result_file1, dialect='excel')
    for row in X_test:
        tem_l1 = row.tolist()
        wr1.writerow(tem_l1)

    """df_original_dataInfo datafram after dividing the aspect of hand column"""

    df_original_dataInfo = making_two_columns(path_original_metadata)
    # print(df_original_dataInfo.head())
    df_original_dataInfo = df_original_dataInfo[['imageName', 'SideOfHand']].copy()
    # print(df_original_dataInfo.head())
    """Merging both dataframes.Important because metadata and folder have different sequence of images"""

    df_unlabelled_Info = pd.merge(df_unlabelled_images_name, df_original_dataInfo, on="imageName")
    df_unlabelled_Info["SideOfHand"] = coding(df_unlabelled_Info["SideOfHand"], {'dorsal': 1, 'palmar': 0})
    """y_test_actual"""

    colCoded = pd.Series(df_unlabelled_Info["imageName"], copy=True)
    for i in range(colCoded.size):
        colCoded[i]= path_unlabelled_images+colCoded[i]
    df_unlabelled_Info["imageName"]= colCoded
    #y_test_actual = df_unlabelled_Info["SideOfHand"]
    df_unlabelled_Info.to_csv('unlabel.csv')
    
    
