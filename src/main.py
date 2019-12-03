from flask import Flask, flash, redirect, render_template, request, session, abort
import src.task2 as t2
import src.task5 as t5
import src.task6_svm as t6_svm
import src.task6_naive as t6_naive
import src.task1 as t1
import src.task4_svm_run as t4svm
import json
from random import shuffle
import os
from src.tasks.task3 import Task3
import glob
from src.models.enums.models import ModelType
from src.tasks.task6PPR import Task6PPR
from src.tasks.task4pprNoCache import Task4PPRNoCache
from src.classifiers.pprClassifier import ImageClass

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/task1/", methods = ['GET', 'POST'])
def task1():
    palmarPath = request.form['palmarPath']
    dorsalPath = request.form['dorsalPath']
    metaDataFile = request.form['metaDataFile']
    inputPath = request.form['inputPath']
    dorsalImages, palmarImages = t1.task1(palmarPath, dorsalPath, metaDataFile, inputPath)
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

@app.route("/task2/", methods = ['GET', 'POST'])
def task2():
    csvpath = request.form['csvpath']
    imagePath = request.form['imagePath']
    queryPath = request.form['queryPath']
    queryCsvPath = request.form['queryCsvPath']

    t2.helper(csvpath, imagePath, queryPath, queryCsvPath)


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

    return render_template('task2.html', dorsalData = dorsalImages, palmarData = palmarImages, queryData = queryImages)

@app.route("/task3/", methods = ['GET', 'POST'])
def task3():
    imageDir = request.form['imageDir']
    q1 = request.form['q1']
    q2 = request.form['q2']
    q3 = request.form['q3']
    queryImagePaths = [q1,q2,q3]
    capitalK = request.form['capitalK']
    smallK = request.form['smallK']
    # imageDir = "static/Dataset2"
    # queryImagePaths = [
    #     "static/Dataset2/Hand_0000010.jpg",
    #     "static/Dataset2/Hand_0000011.jpg",
    #     "static/Dataset2/Hand_0000012.jpg"
    # ]
    # capitalK = 10
    # smallK = 10
    queryImagePathsForRender = [os.path.relpath(imagePath, "static/") for imagePath in queryImagePaths]
    paths = Task3(smallK, imageDir, modelTypes=[ModelType.HOG]).getSimilarImagePaths(capitalK, queryImagePaths)
    pathsToRender = [os.path.relpath(imagePath, "static/") for imagePath in paths]
    return render_template('task3.html', images=pathsToRender, queryImages=queryImagePathsForRender)

@app.route("/task4/svm", methods = ['GET', 'POST'])
def task4_svm():
    print("Boom")
    dorsalImages, palmarImages, accuracy_score = t4svm.helper()
    print("Boom2")
    dorsalImages2, palmarImages2 = [], []
    for img in dorsalImages:
        dorsalImages2.append(getPathForStatic(img))
    for img in palmarImages:
        palmarImages2.append(getPathForStatic(img))
    dorsalImages = dorsalImages2
    palmarImages = palmarImages2
    return render_template("task4_svm.html", dorsalImages = dorsalImages, palmarImages = palmarImages, accuracy_score = accuracy_score)
    return "TASK 4 TO BE DONE"

@app.route("/task4/dt", methods = ['GET', 'POST'])
def task4_dt():
    return "TASK 4 DT TO BE DONE"

@app.route("/task4/ppr", methods = ['GET', 'POST'])
def task4_ppr():
    imgDir = "static/Labelled/Set2"
    csvFile = "static/labelled_set2.csv"
    unlabelledImageDir = "static/Unlabelled/Set 2"

    imageDict = {
        ImageClass.DORSAL.name: [],
        ImageClass.PALMAR.name: []
    }
    for imagePath in glob.glob(os.path.join(unlabelledImageDir, "*.jpg")):
        imageClass = Task4PPRNoCache(imgDir, csvFile, modelTypes=[ModelType.CM]).getClassForImage(imagePath)
        imageDict[imageClass.name].append(imagePath)


    return render_template("task4_ppr.html",
                           dorsalImages=[os.path.relpath(imagePath, "static/") for imagePath in imageDict[ImageClass.DORSAL.name]],
                           palmarImages=[os.path.relpath(imagePath, "static/") for imagePath in imageDict[ImageClass.PALMAR.name]])

@app.route("/task5/", methods = ['GET', 'POST'])
def task5():
    queryImage, candidateImages = t5.helper()
    queryImageName = imageHelper(queryImage)
    queryImage = getPathForStatic(queryImage)
    candidateImagesNames = []
    for img in candidateImages
        candidateImagesNames.append(imageHelper(img))
    candidateImages2 = []
    for img in candidateImages:
        candidateImages2.append(getPathForStatic(img))
    candidateImages = candidateImages2
    print(candidateImages)
    return render_template('task5.html', queryImage = queryImage, queryImageName = queryImageName, candidateImages = candidateImages, candidateImagesNames = candidateImagesNames, len = len(candidateImages))

@app.route("/task6_svm", methods = ['GET', 'POST'])
def task6_svm():
    images = t6_svm.getImages()[:10]
    print(images)
    images = [getPathForStatic(imagePath) for imagePath in images]
    print(images)
    return render_template("task6_svm.html", images=images)

@app.route("/task6_naive", methods = ['GET', 'POST'])
def task6_naive():
    images = t6_naive.getImages()[:10]
    print(images)
    images = [getPathForStatic(imagePath) for imagePath in images]
    print(images)
    return render_template("task6_naive.html", images=images)

@app.route("/process_feedback_naive", methods = ['GET', 'POST'])
def process_feedback_naive():
    data = request.get_json().get('data')
    data['relevant'] = [os.path.abspath('src') + img for img in data['relevant']]
    data['nonrelevant'] = [os.path.abspath('src') + img for img in data['nonrelevant']]
    imagesTemp = t6_naive.main(data)
    print(imagesTemp)
    images = [getPathForStatic(imagePath) for imagePath in imagesTemp]
    return render_template("imageList.html", images = images);


@app.route("/task6_ppr", methods = ['GET', 'POST'])
def task6_ppr():
    images = ['Dataset2/Hand_0006333.jpg',
              'Dataset2/Hand_0006332.jpg',
              'Dataset2/Hand_0006331.jpg',
              'Dataset2/Hand_0000002.jpg',
              'Dataset2/Hand_0000003.jpg',
              'Dataset2/Hand_0000005.jpg',
              'Dataset2/Hand_0000008.jpg']

    return render_template("task6_ppr.html", images=images)

@app.route("/process_feedback_ppr", methods = ['GET', 'POST'])
def process_feedback_ppr():
    # images = ['sample_data/Hands/Hand_0006333.jpg', 'sample_data/Hands/Hand_0006332.jpg','sample_data/Hands/Hand_0006331.jpg','sample_data/Hands/Hand_0000002.jpg','sample_data/Hands/Hand_0000003.jpg','sample_data/Hands/Hand_0000005.jpg','sample_data/Hands/Hand_0000008.jpg']
    # shuffle(images)
    data = request.get_json().get('data')
    imagesTemp = Task6PPR().getRelaventImages(data)
    images = [os.path.relpath(imagePath, "static/") for imagePath in imagesTemp]
    return render_template("imageList.html", images = images);

@app.route("/process_feedback_svm", methods = ['GET', 'POST'])
def process_feedback_svm():
    # images = ['sample_data/Hands/Hand_0006333.jpg', 'sample_data/Hands/Hand_0006332.jpg','sample_data/Hands/Hand_0006331.jpg','sample_data/Hands/Hand_0000002.jpg','sample_data/Hands/Hand_0000003.jpg','sample_data/Hands/Hand_0000005.jpg','sample_data/Hands/Hand_0000008.jpg']
    # shuffle(images)
    data = request.get_json().get('data')
    data['relevant'] = [os.path.abspath('src') + img for img in data['relevant']]
    data['nonrelevant'] = [os.path.abspath('src') + img for img in data['nonrelevant']]
    imagesTemp = t6_svm.helper(data)
    images = [getPathForStatic(imagePath) for imagePath in imagesTemp]
    return render_template("imageList.html", images = images);

if __name__ == "__main__":
    app.run()