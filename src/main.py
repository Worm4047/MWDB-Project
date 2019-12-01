from flask import Flask, flash, redirect, render_template, request, session, abort
import src.task2 as t2
import src.task5 as t5
import src.task1 as t1
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
    dorsalImages, palmarImages = t1.task1()
    return render_template('index.html',  dorsalImages = dorsalImages, palmarImages = palmarImages)


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
    # t2.helper()
    dorsalImages = getLabelledImages(True)
    palmarImages = getLabelledImages(False)
    queryImages = getQueryImageResuls()
    queryImages['PALMAR'] = reversed(queryImages['PALMAR'])
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

@app.route("/task4/svm")
def task4_svm():
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

@app.route("/task4/dt")
def task4_dt():
    return "TASK 4 DT TO BE DONE"

@app.route("/task4/ppr")
def task4_ppr():
    return "TASK 4 PPR TO BE DONE"

@app.route("/task5/")
def task5():
    queryImage, candidateImages = t5.helper()
    queryImageName = imageHelper(queryImage)
    queryImage = getPathForStatic(queryImage)
    candidateImagesNames = []
    for img in candidateImages:
        candidateImagesNames.append(imageHelper(img))
    candidateImages2 = []
    for img in candidateImages:
        candidateImages2.append(getPathForStatic(img))
    candidateImages = candidateImages2
    print(candidateImages)
    return render_template('task5.html', queryImage = queryImage, queryImageName = queryImageName, candidateImages = candidateImages, candidateImagesNames = candidateImagesNames, len = len(candidateImages))

@app.route("/task6/")
def task6():
    images = ['sample_data/Hands/Hand_0006333.jpg', 'sample_data/Hands/Hand_0006332.jpg','sample_data/Hands/Hand_0006331.jpg','sample_data/Hands/Hand_0000002.jpg','sample_data/Hands/Hand_0000003.jpg','sample_data/Hands/Hand_0000005.jpg','sample_data/Hands/Hand_0000008.jpg']
    return render_template("task6.html", images = images)

@app.route("/process_feedback", methods = ['GET', 'POST'])
def process_feedback():
    images = ['sample_data/Hands/Hand_0006333.jpg', 'sample_data/Hands/Hand_0006332.jpg','sample_data/Hands/Hand_0006331.jpg','sample_data/Hands/Hand_0000002.jpg','sample_data/Hands/Hand_0000003.jpg','sample_data/Hands/Hand_0000005.jpg','sample_data/Hands/Hand_0000008.jpg']
    shuffle(images)
    data = request.get_json().get('data')
    print(data)
    return render_template("imageList.html", images = images);
if __name__ == "__main__":
    app.run()