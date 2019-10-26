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

start_time = time.time()

def sort_print_n_return(npArray):
    ind = np.flip(np.argsort(npArray, axis=1), 1)
    rows = ind.shape[0]
    cols = ind.shape[1]
    twpair = np.zeros((cols,), dtype = 'i,f')
    for x in range(0, rows):
        l = []
        for y in range(0, cols):
            l.append((ind[x, y], npArray[x,ind[x,y]]))
        twpair = np.vstack([twpair,  np.array(l, dtype='i,f')])
    #need to delete the initialized all zero row
    twpair = np.delete(twpair, (0), axis=0)
    return twpair

def prepros(csvFilePath, databasePath, destpath):
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
        mat = (getDataMatrix(None, ModelType.LBP, label=None, directoryPath = destpath))
        u,vt = SVD(mat, minu).getDecomposition()
        #vt = vt.tolist()

        print("Got decomp for :",k)
        dic1[k]=vt

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
    if not os.path.exists(destpath):
        os.makedirs(destpath)
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
    u,v = NMF(m_np, k).getDecomposition()
    t = sort_print_n_return(v)
    for i in range(len(t)):
        print("Latent Semantic ",i+1,":")
        for j in range(len(t[i])):
            k = t[i][j][0]
            print("Subject Id:",key[k],"\t Weight:",t[i][j][1])

print("Execution Time :",( time.time()-start_time))
