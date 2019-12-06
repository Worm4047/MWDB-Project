import numpy as np
from random import randrange
import csv
import math
import random
from csv import reader
import pickle
from src.dimReduction.dimRedHelper import DimRedHelper
from sklearn.decomposition import PCA
from src.common.imageHelper import ImageHelper
from src.constants import BLOCK_SIZE
from src.models.ColorMoments import ColorMoments
def load_csv_dataset(filename):
    """Load the CSV file"""
    file = open(filename, "r")
    lines = reader(file)
    dataset = [list(map(float, row)) for row in lines]
    return dataset


def getImages():
    li = []
    with open('store/ls_image.pkl', 'rb') as f2:
        li = pickle.load(f2)    
    return li

def getCandiddateImages():
    li = []
    try:
        with open('store/lsh_candidate_images.pkl', 'rb') as f2:
            li = pickle.load(f2)
    except:
        pass
    return li

def mean(numbers):
    """Returns the mean of numbers"""
    return np.mean(numbers)


def stdev(numbers):
    """Returns the std_deviation of numbers"""
    return np.std(numbers)


def sigmoid(z):
    """Returns the sigmoid number"""
    return 1.0 / (1.0 + math.exp(-z))


def cross_validation_split(dataset, n_folds):
    """Split dataset into the k folds. Returns the list of k folds"""
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, dt_test):
    """Evaluate an algorithm using a cross validation split"""
    predicted = algorithm(dataset, dt_test, )
    return predicted


#############################
#############################
######## Naive Bayes  #######
#############################
#############################


def separate_by_class(dataset):
    """Split training set by class value"""
    separated = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)
    return separated


def model(dataset):
    """Find the mean and standard deviation of each feature in dataset"""
    # models = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    models = []
    stds = np.std(dataset, axis=0)
    # print("Standard deviation",stds)
    means = np.mean(dataset, axis = 0)

    for i in range(len(stds)):
        models.append((means[i], stds[i]))
        # print(means[i], stds[i])

    # print(models)
    models.pop() #Remove last entry because it is class value.
    return models


def model_by_class(dataset):
    """find the mean and standard deviation of each feature in dataset by their class"""
    separated = separate_by_class(dataset)
    class_models = {}
    for (classValue, instances) in separated.items():
        class_models[classValue] = model(instances)
    
    return class_models


def calculate_pdf(x, mean, stdev):
    """Calculate probability using gaussian density function"""
    if stdev == 0.0:
        # print("STD IS ZERO")
        if x == mean:
            return 1.0
        else:
            return 0.0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent


def calculate_class_probabilities(models, input):
    """Calculate the class probability for input sample. Combine probability of each feature"""
    probabilities = {}
    for (classValue, classModels) in models.items():
        probabilities[classValue] = 1
        # print(len(classModels), len(input))
        for i in range(len(classModels)):
            (mean, stdev) = classModels[i]
            x = input[i]
            v = calculate_pdf(x, mean, stdev)
            if v != 0.0:
                # print(probabilities[classValue])
            # print(v, probabilities[classValue])
                probabilities[classValue] *= v
    return probabilities


def predict(models, inputVector):
    """Compare probability for each class. Return the class label which has max probability."""
    probabilities = calculate_class_probabilities(models, inputVector)
    # print("Probabilities", probabilities)
    (bestLabel, bestProb) = (None, -1)
    for (classValue, probability) in probabilities.items():
        if bestLabel is 1 or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
        # print(bestProb, bestLabel)  
    return bestProb


def getPredictions(models, testSet):
    """Get class label for each value in test set."""
    predictions = []
    for i in range(len(testSet)):
        result = predict(models, testSet[i])
        predictions.append(result)
    return predictions


def naive_bayes(train, test, ):
    """Create a naive bayes model. Then test the model and returns the testing result."""
    summaries = model_by_class(train)
    predictions = getPredictions(summaries, test)
    return predictions


def main(d):

    liImages = []
    labels = []
    t = 10
    # print(d)
    try:
        # with open('src/store/lsh_candidate_images.pkl', 'rb') as f2:
        #     temp = pickle.load(f2)
        #     t = len(temp)
        with open('store/task6_naive_iteration_images.pkl', 'rb') as f2:
            liImages = pickle.load(f2) 
        with open('store/task6_naive_iteration_labels.pkl', 'rb') as f2:
            labels = pickle.load(f2) 
    except:
        pass
    print(len(liImages))
    for img in d['relevant']:
        if img not in liImages:
            labels.append(1)
            liImages.append(img)

        else:
            indx = liImages.index(img)
            labels[indx] = 1

    for img in d['nonrelevant']:
        if img not in liImages:
            liImages.append(img)
            labels.append(0)
        else:
            indx = liImages.index(img)
            labels[indx] = 0
    print(len(liImages))
    # print(labels)
    with open('store/task6_naive_iteration_images.pkl', 'wb') as f2:
        pickle.dump(liImages, f2)

    with open('store/task6_naive_iteration_labels.pkl', 'wb') as f2:
        pickle.dump(labels, f2)

    names = getImages()
    # names = list(set(names) - set(liImages))

    obj = DimRedHelper()
    # print("Labels len {} images len {} names len{}".format(len(liImages), len(labels), len(names)))
    # dt_test = []
    # dataset = []
    # obj = ImageHelper()
    # for img in names:
    #     print(img)
    #     feature_vector = ColorMoments(obj.getYUVImage(img), BLOCK_SIZE, BLOCK_SIZE).getFeatures()
    #     dt_test.append(feature_vector)
    # # print(liImages)
    # for img in liImages:
    #     feature_vector = ColorMoments(obj.getYUVImage(img), BLOCK_SIZE, BLOCK_SIZE).getFeatures()
    #     dataset.append(feature_vector)

    dt_test = obj.getDataMatrixForHOG(names, [])
    dataset = obj.getDataMatrixForHOG(liImages, [])

    pca = PCA(n_components=7)
    dataset = (pca.fit_transform(dataset))
    # pca = PCA(n_components=5)
    dt_test = (pca.transform(dt_test))

    dataset2 = []  
    dt_test2 = []  
    # print(dt_test[0].shape, dataset[0].shape)
    for i in range(len(dt_test)):
        dt_test2.append(np.append(dt_test[i], None))
    for i in range(len(liImages)):
        # print(dataset[i].shape, type(dataset[i]))
        dataset2.append(np.append(dataset[i], labels[i]))
    # print(dataset2)
    #load and prepare data
    # getdata()
    # dataset = load_csv_dataset('train_data.csv')
    # random.shuffle(dataset)
    # test_dataset='test_data.csv'
    # dt_test= load_csv_dataset(test_dataset)

    # file = open('unlabel.csv', "r")
    # lines = reader(file)
    # test_label = [list(row) for row in lines]
    # actual = []
    # for i in range(1,len(test_label)):
    #     actual.append(test_label[i][2])

    n_folds = 3
    print ("---------- Gaussian Naive Bayes ---------------")
    predictions_probability = evaluate_algorithm(dataset2, naive_bayes, n_folds, dt_test2)
    li = []
    for i in range(len(names)):
        li.append((predictions_probability[i], names[i]))
    res = sorted(li, key=lambda tup: tup[0])
    res.reverse()
    print(res)
    images = []
    for elem in res:
        images.append(elem[1])
    # print(images)
    return images[:t]

# if __name__ == '__main__':
#     d = {'relevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006333.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006332.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000005.jpg'], 'nonrelevant': ['/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0006331.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000002.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000003.jpg', '/home/worm/Desktop/ASU/CSE_515/MWDB-Project/src/static/sample_data/Hands/Hand_0000008.jpg']}
#     main(d)