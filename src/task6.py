import pandas as pd
import os
import shutil
from dimReduction.dimRedHelper import getDataMatrix
from dimReduction.PCA import calcPCA
from models.enums.models import ModelType
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
from dimReduction.SVD import SVD

def cos(a,b):
    cos_lib = cosine_similarity(a, b)
    cos_l = np.mean(cos_lib)
    return cos_l
def task6(id):
    csvFilePath = "/Users/user/Documents/HandInfo.csv"
    databasePath = "/Users/user/Documents/Hands"
    destpath = "/Users/user/Documents/Task6"
    df = pd.read_csv(csvFilePath, usecols = ['id','imageName'])
    dic = dict()
    for r, i in df.iterrows():
        j = i[0]
        l = i[1]
        if j in dic:
            dic[j].append(l)
        else :
            dic.setdefault(j, [])
            dic[j].append(l)
    dic1 = dict()
    for k, v in dic.items():
        for imageName in v:
            shutil.copy(os.path.join(databasePath, imageName), destpath)
        mat = (getDataMatrix(None, ModelType.CM, directoryPath ="/Users/user/Documents/Task6"))
        u,vt =calcPCA((mat), 10)
        if(k==id):
            queryrep = vt
        else:
            dic1.setdefault(k, [])
            dic1[k].append(vt)
        for imageName in v:
                temp = destpath+"/"+imageName
                os.remove(temp)
    #for k, v in dic1.items():
        #print("Key:",k)
    for k, v in dic1.items():
        sim = cos(queryrep,v)
        dic1[k]=sim
    sort_d = sorted(dic1, key = lambda x:x[1])
    print ("Most related 3 subjects for Subject ",id," are :")
    f = 1
    for k, v in dic1.items():
        print("1st Subject :",k, "    Similarity:",v)
        if(f==3):
            break
        f+=1

task6(4)
    

