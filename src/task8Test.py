import pandas as pd
import numpy as np
import glob
import os
import csv
from src.dimReduction.NMF import NMF
from src.common.sort_print_n_return_desc import sort_print_n_return


def making_two_columns(file_HandInfo):
    df_hands_info=pd.read_csv(file_HandInfo, delimiter=",")
    new = df_hands_info["aspectOfHand"].str.split(" ", n=1, expand=True)
    df_hands_info["SideOfHand"] = new[0]
    df_hands_info["WhichHand"] = new[1]
    #df_hands_info.drop(columns =["aspectOfHand"], inplace = True)
    return df_hands_info

def coding(col, codedict):
    colcoded = pd.Series(col, copy=True)
    for key, value in codedict.items():
        colcoded.replace(key, value, inplace=True)
    return colcoded

def csv_hand_names(HandInfoCSV,imageDir):
    with open(HandInfoCSV + "table_for_Hand_names.csv", mode='w+') as table_for_Hand_names:
        table_for_Hand_names_writer = csv.writer(table_for_Hand_names, delimiter=',', quotechar='"',
                                                 quoting=csv.QUOTE_MINIMAL)
        for filename in glob.glob(imageDir + "*.jpg"):
            file_name = os.path.basename(filename)
            # file_name=file_name.replace(".txt",".jpg")
            table_for_Hand_names_writer.writerow([file_name])


    df_Hand_names = pd.read_csv(HandInfoCSV + "table_for_Hand_names.csv", delimiter=",", header=None)
    df_Hand_names.columns = ["ImageName"]
    return df_Hand_names

def initTask8(imageDir, handInfoCSV, k):

    df_Hand_names=csv_hand_names(handInfoCSV,imageDir)
    df_hands_info = making_two_columns(handInfoCSV + "HandInfo.csv")
    df_Hand_names_Specific = df_hands_info[['imageName', 'gender', 'accessories', 'SideOfHand', 'WhichHand']].copy()
    #print(df_Hand_names_Specific.head())
    df_Hands_Image_MetaData = df_Hand_names.merge(df_Hand_names_Specific, left_on='ImageName', right_on='imageName',
                                                  how='left')
    df_Hands_Image_MetaData.drop(columns=['imageName'], inplace=True)
    df_Hands_Image_MetaData.rename(
        columns={"gender": "male", "accessories": "with accessories", "SideOfHand": "Dorsal", "WhichHand": "Right"},
        inplace=True)

    #### Coding nominal data
    df_Hands_Image_MetaData["male"] = coding(df_Hands_Image_MetaData["male"], {'male': 1, 'female': 0})
    df_Hands_Image_MetaData["Dorsal"] = coding(df_Hands_Image_MetaData["Dorsal"], {'dorsal': 1, 'palmar': 0})
    df_Hands_Image_MetaData["Right"] = coding(df_Hands_Image_MetaData["Right"], {'right': 1, 'left': 0})

    ###Adding other 4 columns
    new_col = (df_Hands_Image_MetaData["male"] - 1).abs()
    df_Hands_Image_MetaData.insert(2, column='female', value=new_col)
    new_col = (df_Hands_Image_MetaData["with accessories"] - 1).abs()
    df_Hands_Image_MetaData.insert(4, column='without accessories', value=new_col)
    new_col = (df_Hands_Image_MetaData["Dorsal"] - 1).abs()
    df_Hands_Image_MetaData.insert(6, column='Palmar', value=new_col)
    new_col = (df_Hands_Image_MetaData["Right"] - 1).abs()
    df_Hands_Image_MetaData.insert(8, column='Left', value=new_col)

    # Call NMF on binary image data matrix
    print(df_Hands_Image_MetaData.iloc[:, 1:9].to_numpy())
    W, H = NMF(df_Hands_Image_MetaData.iloc[:, 1:9].to_numpy(), k, None, 0.0001, 200).getDecomposition()
    print('W:', W)
    print('H:', H)

    # Call To return term weight pairs
    twpair_metadata=sort_print_n_return(H)
    twpair_image=sort_print_n_return(W.transpose())

    return twpair_metadata, twpair_image

if __name__ == "__main__":
    imageDir="D:/studies/multimedia and web databases/project/CSE 515 Fall19 - Smaller Dataset\CSE 515 Fall19 - Smaller Dataset/"
    handInfoCSV="D:/studies/multimedia and web databases/project/"
    k=5
    twpair_metadata, twpair_image= initTask8(imageDir, handInfoCSV, k)
    print( "Term weight pair data for image:",twpair_image)
    print("Term weight pair for metadata:",twpair_metadata)