# Introduction

Given a query image, the project aims at efficiently classifying or searching for similar images in a data-set of images. This phase of project focuses on implementation of different clustering, indexing and classification techniques. Personalised page rank algorithm is used to find the most dominant images in the database with respect to the images given as input. A relevance feedback system is implemented on top of the similar images retrieved using LSH based on each classification technique to improve the accuracy of the results.
The data set used was provided by https://sites.google.com/view/11khands, M. Multimed Tools Appl (2019) 78: 20835

Multimedia data is being generated exponentially from various domains, social media being the prime source.  This influx of huge amount of data imposes many challenges in storing, maintaining  and  retrieving  them  as  per  the  requirement. Making  use  of  this  data  and discovering  useful  patterns  out  of  them  has  become  an  active  research  area  in  the  recent times.

Images are the most common type of media found on various platforms and they are generally represented by a 2-dimensional vector.  This vector representation makes it a difficult task for a computer to understand the visual data which humans can easily understand by just looking at an image.  Directly operating on this vector could also be a very expensive and a time consuming task. Given this matrix to a computer,  extracting meaningful and high level information for that image has been a major challenge.

Classification of data is explored by experimenting with our own implementations of different classification techniques like SVM, Decision Trees to label an input image as dorsal or palmar. A personalized version of page rank is implemented to find the ranks or relevance of images in the database with respect to other images. Partitioning and Indexing data ensures an efficient retrieval of similar images. While indexing RDBMS requires only a simple sorting, in the case of multimedia database, it wont be sufficient. We explore LSH as an indexing technique and spectral clustering for creating clusters in the image set.

## Installation

Use package pip or create a venv to install the following libraries.


```bash
pip install pandas
pip install numpy
pip install scipy
pip install scikit−learn
pip install matplotlib
pip install tkinter
```

## Usage

```python
python -m src.testInterface.py
```
After running the above command, follow the instructions on the user interface to execute the required functionality.
