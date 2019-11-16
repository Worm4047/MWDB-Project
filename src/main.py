from flask import Flask, flash, redirect, render_template, request, session, abort
from src.task2 import helper
import json
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

@app.route("/task2/")
def task2():
    # helper()
    dorsalImages = getLabelledImages(True)
    palmarImages = getLabelledImages(False)
    queryImages = getQueryImageResuls()
    print(type(queryImages['PALMAR'].reverse()))
    # print(dorsalImages)
    # print(palmarImages)
    # return "Task2"
    return render_template('task2.html', dorsalData = dorsalImages, palmarData = palmarImages, queryData = queryImages)

@app.route("/task3/")
def task3():
    return "TASK 3 TO BE DONE"       

@app.route("/task4/")
def task4():
    return "TASK 4 TO BE DONE"

@app.route("/task5/")
def task5():
    return "TASK 5 TO BE DONE"

@app.route("/task6/")
def task6():
    return "TASK 6 TO BE DONE"
if __name__ == "__main__":
    app.run()