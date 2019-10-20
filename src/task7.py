import pandas as pd
import os
import shutil
from src.dimReduction.dimRedHelper import getDataMatrix
from src.dimReduction.PCA import calcPCA
from src.models.enums.models import ModelType
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
from src.dimReduction.NMF import NMF
import time

start_time = time.time()

def prepros(csvFilePath, databasePath, destpath):
    #csvFilePath = "/Users/user/Documents/HandInfo.csv"
    #databasePath = "/Users/user/Documents/Hands"
    #destpath = "/Users/user/Documents/Task6"
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

def task7(k, csvFilePath, databasePath, destpath, task7_file):
    if(os.stat("file").st_size == 0):
        dictn1 = prepros(csvFilePath, databasePath, destpath)
    else:
        with open(task7_file,'w') as file:
            dictn1 = json.loads(file.write())
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
    print("Subject Subject Similarity Matrix:")
    print(df)
    m_np = np.array(m_out)
    u,v = NMF(m_np, k).getDecomposition()

#task7(10)
print("Execution Time :",( time.time()-start_time))

    