import pandas as pd
import os
import shutil
from src.dimReduction.dimRedHelper1 import getDataMatrix
from src.dimReduction.PCA import calcPCA
from src.models.enums.models import ModelType
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import glob
from src.dimReduction.NMF import NMF
import time
import json

from src.common import util

start_time = time.time()

def prepros(csvFilePath, databasePath, destpath):
    #csvFilePath = "/Users/user/Documents/HandInfo.csv"
    #databasePath = "/Users/user/Documents/Hands"
    #destpath = "/Users/user/Documents/Task6"
    df = pd.read_csv(csvFilePath, usecols = ['id','imageName'])
    onlyfiles = [f for f in listdir(databasePath) ]
    dic = dict()
    min_d = dict()
    flag=0
    minu=0
    for r, i in df.iterrows():
        #print(type(i[1]))
        j = i[0]
        l = i[1]
        if l in onlyfiles:
            if j in dic:
                dic[j].append(l)
                min_d[j]+=1
            else :
                dic.setdefault(j,[])
                dic[j].append(l)
                min_d[j]=1
        

    #print(dic)
    for k,v in min_d.items():
        print(k)
        print("no. of image:", v)
        if(flag==0):
            minu=v
            flag=1
        else:
            if minu>v:
                minu =v
    print(minu)
    if not os.path.exists(destpath):
        os.makedirs(destpath)

        
    dic1 = dict()
    for k, v in dic.items():
        for imageName in v:
            shutil.copy(os.path.join(databasePath, imageName), destpath)
        print("just:",k)
        mat = (getDataMatrix(None, ModelType.CM, label=None, directoryPath = destpath))
        #print("Got matrix for ",k)
        #print(mat)
        #print("-------------------------------------------------------------------------------------------")

        u,vt = PCA(mat, minu).getDecomposition()
        #vt = vt.tolist()

        print("Got decomp for :",k)
        dic1[k]=vt
            #print(dic1[k])
            #print(vt)
        for imageName in v:
                temp = destpath+"/"+imageName
                os.remove(temp)
    return dic1

def cos(a,b):
    cos_lib = euclidean_distances(a, b)
    cos_l = np.mean(cos_lib)
    return cos_l

def task7(k, csvFilePath, databasePath, destpath, filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = os.path.join(filepath,"ssmatrix.xls")
    if not os.path.exists(filename):
        open(filename, 'w').close()
    task7_file = os.path.join(filepath,"Task7_input.txt")
    if(os.stat(task7_file).st_size == 0):
        dictn1 = prepros(csvFilePath, databasePath, destpath)
        
    else:
        with open(task7_file,'r') as file:
            dictn1 = json.loads(file.read())
    key = [*dictn1]
    val = [*dictn1.values()]
    m_out = [[0 for i in range(len(key))] for j in range(len(key))]


    for i in range(len(val)):
        m_out[i][i]=0
        for j in range(i+1,len(val)):
            sim_t = cos(val[i],val[j])
            m_out[i][j]=sim_t
            m_out[j][i]=sim_t
    df = pd.DataFrame(m_out, index=key, columns=key)
    df.to_excel(filename)
    print("Subject Subject Similarity Matrix:")
    print(df)
    m_np = np.array(m_out)
    #u,v = NMF(m_np, k).getDecomposition()
    #util.sort_print_n_return(v)

task7(10, "/Users/user/Documents/HandInfo.csv", "/Users/user/Documents/Hands2", "/Users/user/Documents/Task6", "/Users/user/Documents")
print("Execution Time :",( time.time()-start_time))

    