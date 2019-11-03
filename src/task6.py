import pandas as pd
import os
import shutil
from src.dimReduction.dimRedHelper1 import getDataMatrix
from src.dimReduction.PCA import PCA
from src.dimReduction.SVD import SVD
from src.models.enums.models import ModelType
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time
from operator import itemgetter  
import json
from os import listdir
from src.common import plotHelper
import cv2


start_time = time.time()

def cos(a,b):
    cos_lib = euclidean_distances(a, b)
    cos_l = np.mean(cos_lib)
    return cos_l

def task6(id, csvFilePath, databasePathid, databasePath, destpath, filepath, ):
    #destpath = "/Users/user/Documents/Task6"
    df = pd.read_csv(csvFilePath, usecols = ['id','imageName'])
    onlyfiles = [f for f in listdir(databasePath) ]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    task7_file = os.path.join(filepath,"Task7_input.txt")
    if not os.path.exists(task7_file):
        open(task7_file, 'w').close()
    dic = dict()
    min_d = dict()
    flag=0
    minu=0
    for r, i in df.iterrows():
        #print(type(i[1]))
        j = i[0]
        l = i[1]
        if l in onlyfiles or j==id:
            if j in dic:
                dic[j].append(l)
                min_d[j]+=1
            else :
                dic.setdefault(j,[])
                dic[j].append(l)
                min_d[j]=1

        

    #print(dic)
    for k,v in min_d.items():
        #print(k)
        #print("no. of image:", v)
        if(flag==0):
            minu=v
            flag=1
        else:
            if minu>v:
                minu =v
    #print(minu)
    if not os.path.exists(destpath):
        os.makedirs(destpath)

        
    dic1 = dict()
    for k, v in dic.items():
        if(k==id):
            for imageName in v:
                shutil.copy(os.path.join(databasePathid, imageName), destpath)
        else:
            for imageName in v:
                shutil.copy(os.path.join(databasePath, imageName), destpath)
        #print("just:",k)
        mat = (getDataMatrix(None, ModelType.LBP, label=None, directoryPath = destpath))
        #print("Got matrix for ",k)
        #print(mat)
        #print("-------------------------------------------------------------------------------------------")

        u,vt = SVD(mat, minu).getDecomposition()
        vt = vt.tolist()

        print("Got decomposition for :",k)
        if(k==id):
            queryrep = vt
        else:
            dic1[k]=vt
        #print("v transpose: ",vt)
            #print(dic1[k])
            #print(vt)
        for imageName in v:
                temp = destpath+"/"+imageName
                os.remove(temp)
    """for k,v in dic1.items():
        print(k)
        print(v)"""
        #print("-----------------------------------------------------------------------------------------------")
    with open(task7_file,'w') as file:
        file.write(json.dumps(dic1))
    #for k, v in dic1.items():
        #print("Key:",k)
    for k, v in dic1.items():
        sim = cos(queryrep,v)
        dic1[k]=sim
        print("Got similarity for:",k)
    #print(dic1)    
    dic2 = dict()
    sort_d = sorted(dic1.items(), key = itemgetter(1), reverse=True)
    lis = dic[id]
    for l in lis:
        if l not in dic2:
            dic2[l] = cv2.cvtColor(cv2.imread(databasePathid+"/"+l, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    plotHelper.plotFigures(dic2)
    print ("Most related 3 subjects for Subject ",id," are :")
    f = 1
    for k, v in sort_d:

        print(" Subject ",f," :",k, "    Similarity:",v)
        lis = dic[k]
        dic3 = dict()
        for l in lis:
            if l not in dic3:
                dic3[l]= cv2.cvtColor(cv2.imread(databasePath+"/"+l, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        plotHelper.plotFigures(dic3)
        if(f==3):
            break
        f+=1
