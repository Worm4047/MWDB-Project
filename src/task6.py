import pandas as pd
import os
import shutil
from src.dimReduction.dimRedHelper import getDataMatrix
from src.dimReduction.PCA import PCA
from src.models.enums.models import ModelType
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
from src.dimReduction.SVD import SVD
import time
from operator import itemgetter  

start_time = time.time()

def cos(a,b):
    cos_lib = cosine_similarity(a, b)
    cos_l = np.mean(cos_lib)
    return cos_l
def task6(id, csvFilePath, databasePath):
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
    #print(minu)
        
    dic1 = dict()
    for k, v in dic.items():
        for imageName in v:
            shutil.copy(os.path.join(databasePath, imageName), destpath)
        mat = (getDataMatrix(None, ModelType.CM, directoryPath ="/Users/user/Documents/Task6"))

        u,vt = PCA(mat, minu).getDecomposition()
        print("k :",k,"vt shape:",vt.shape)
        if(k==id):
            queryrep = vt
        else:
            dic1[k]=vt
            #print(dic1[k])
            #print(vt)
        for imageName in v:
                temp = destpath+"/"+imageName
                os.remove(temp)
    #for k, v in dic1.items():
        #print("Key:",k)
    for k, v in dic1.items():
        sim = cos(queryrep,v)
        dic1[k]=sim
    sort_d = sorted(dic1.items(), key = itemgetter(1), reverse=True)
    print ("Most related 3 subjects for Subject ",id," are :")
    f = 1
    for k, v in sort_d:
        print(" Subject f :",k, "    Similarity:",v)
        if(f==3):
            break
        f+=1

#task6(10)
print("Execution Time :",( time.time()-start_time))