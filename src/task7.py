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
import time
from operator import itemgetter  

start_time = time.time()

def prepros():
    csvFilePath = "/Users/user/Documents/HandInfo.csv"
    databasePath = "/Users/user/Documents/Hands"
    destpath = "/Users/user/Documents/Task6"
    df = pd.read_csv(csvFilePath, usecols = ['id','imageName'])
    dic = dict()
    min_d = dict()
    minu =999999
    for r, i in df.iterrows():
        j = i[0]
        l = i[1]
        if j in dic:
            dic[j].append(l)
            min_d[j]+=1
        else :
            dic.setdefault(j,[])
            dic[j].append(l)
            min_d[j]=1
    for k,v in min_d.items():
        if minu>v:
            minu =v     
    dic1 = dict()
    for k, v in dic.items():
        for imageName in v:
            shutil.copy(os.path.join(databasePath, imageName), destpath)
        mat = (getDataMatrix(None, ModelType.CM, directoryPath ="/Users/user/Documents/Task6"))

        u,vt =calcPCA(mat, minu)
        dic1[k]=vt
        for imageName in v:
                temp = destpath+"/"+imageName
                os.remove(temp)
    return dic1

def cos(a,b):
    cos_lib = cosine_similarity(a, b)
    cos_l = np.mean(cos_lib)
    return cos_l

def task7():
    dictn1 = prepros()
    key = [*dictn1]
    val = [*dictn1.values()]
    m_out = [[0 for i in range(len(key))] for j in range(len(key))]


    for i in range(len(val)):
        print(i)
        m_out[i][i]=1
        for j in range(i+1,len(val)):
            sim_t = cos(val[i],val[j])
            m_out[i][j]=sim_t
            m_out[j][i]=sim_t
    df = pd.DataFrame(m_out, index=key, columns=key)
    print(df)

task7()
print("Execution Time :",( time.time()-start_time))

    