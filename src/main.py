from flask import Flask, flash, redirect, render_template, request, session, abort
from src.task2 import helper
import src.task5 as t5
import src.task4_run as t4svm
import json
from random import shuffle
import os
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/task1/")
def task1():
    return "TASK 1 TO BE DONE"


def getLabelledImages(dorsal):
    path = ''

    if(dorsal):
        path = 'src/store/dorsal_labels.json'
    else:
        path = 'src/store/palmar_labels.json'
    print(path)
    with open(path ) as json_file:
        data = json.load(json_file)
        return data

def getQueryImageResuls():
    path = 'src/store/query_labels.json'
    with open(path ) as json_file:
        data = json.load(json_file)
        return data    

def imageHelper(img):
    imagename = os.path.basename(img)
    return imagename

def getPathForStatic(img):
    idx = img.find('/static/')
    res = img[idx+8:]
    return res

@app.route("/task2/")
def task2():
    # helper()
    dorsalImages = getLabelledImages(True)
    palmarImages = getLabelledImages(False)
    queryImages = getQueryImageResuls()
    shuffle(queryImages['PALMAR'])
    for key in dorsalImages:
        li = []
        for elem in dorsalImages[key]:
            imageName, imagePath = elem[0], elem[1]
            imagePath = getPathForStatic(imagePath)
            li.append([imageName, imagePath])
        dorsalImages[key] = li
    for key in palmarImages:
        li = []
        for elem in palmarImages[key]:
            imageName, imagePath = elem[0], elem[1]
            imagePath = getPathForStatic(imagePath)
            li.append([imageName, imagePath])
        palmarImages[key] = li
    for key in queryImages:
        li = []
        for elem in queryImages[key]:
            imageName, imagePath = elem[0], elem[1]
            imagePath = getPathForStatic(imagePath)
            li.append([imageName, imagePath])
        queryImages[key] = li
    # print(dorsalImages)
    # print(palmarImages)
    # return "Task2"
    return render_template('task2.html', dorsalData = dorsalImages, palmarData = palmarImages, queryData = queryImages)

@app.route("/task3/")
def task3():
    return "TASK 3 TO BE DONE"       

@app.route("/task4/")
def task4():
    dorsalImages, palmarImages, accuracy_score = t4svm.helper()
    dorsalImages2, palmarImages2 = [], []
    for img in dorsalImages:
        dorsalImages2.append(getPathForStatic(img))
    for img in palmarImages:
        palmarImages2.append(getPathForStatic(img))
    dorsalImages = dorsalImages2
    palmarImages = palmarImages2
    return render_template("task4_svm.html", dorsalImages = dorsalImages, palmarImages = palmarImages, accuracy_score = accuracy_score)
    return "TASK 4 TO BE DONE"

@app.route("/task5/")
def task5():
    queryImage, candidateImages = t5.helper()
    queryImageName = imageHelper(queryImage)
    queryImage = getPathForStatic(queryImage)
    candidateImagesNames = set()
    for img in candidateImages:
        candidateImagesNames.add(imageHelper(img))
    candidateImages2 = set()
    for img in candidateImages:
        candidateImages2.add(getPathForStatic(img))
    candidateImages = list(candidateImages2)
    print(candidateImages)
    return render_template('task5.html', queryImage = queryImage, queryImageName = queryImageName, candidateImages = candidateImages, candidateImagesNames = candidateImagesNames, len = len(candidateImages))

@app.route("/task6/")
def task6():
    return "TASK 6 TO BE DONE"
if __name__ == "__main__":
    app.run()