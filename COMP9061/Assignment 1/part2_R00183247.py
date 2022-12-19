
from mpmath import nint, sqrt
import numpy as np
from collections import Counter
import sys

#method for loading datasets
def loadData(file):
    #Load data from the file
    data = np.loadtxt(open(file, "rb"), delimiter=",")
    return data

# method for calculating distance
#featureData - numpy array with feature vectors
#queryInstance - the vector to calculate distance from
#returns distances array and array of indexes sorted by shotest distance
def calculateDistances(featureData, queryInstance):
    #calculating differences element wise (qn-pn)
    ab = featureData - queryInstance
    #Perfomring second part of calculation square, sum and square root
    #Using Einstein summation convention: "ij, ij" will perform a square item wise "->" will sum the outputs
    #this can as well be achirved using np.sqrt, np.square and sum 
    ed = np.sqrt(np.einsum('ij,ij->i',ab, ab))
    #return the distances and sorted indexes
    return ed, np.argsort(ed)

def calculateMinkowski(featureData, queryInstance, n):
    ab = np.power(np.absolute( featureData - queryInstance), n)
    md = np.power(np.sum(ab, axis=1), 1/n)
    return md, np.argsort(md)


def processClasses(trainData, indexes, distances, n):
    #This may cause division by 0 if the distance is 0
    wDistances = 1 / np.power(distances, n)
    temp = np.zeros(shape=(indexes.shape[0],2))
    for i in range(indexes.shape[0]):
        temp[i] = [trainData[:,10][indexes[i]], wDistances[indexes[i]]]
    #calculate the weighted values by classes. In this scenario I have only 3 classes: 0, 1, 2
    weightedClass = np.array([np.sum(temp[temp[:,0] == 0][:,1]),np.sum(temp[temp[:,0] == 1][:,1]), np.sum(temp[temp[:,0] == 2][:,1]) ])
    return weightedClass

def scale(trainDataset, testDataset):
    max = np.max(trainDataset, axis=0)
    min = np.min(trainDataset, axis=0)
    bot = max - min
    scale = (trainDataset - min)
    s2 = np.divide(scale,bot)
    s3 = np.divide((testDataset - min), bot) 
    return s2, s3


#method for finding k-NN using Minkowski
#trainData - the train dataset including features and classes
#testData - the test dataset including features and classes
#k - the numer of nearest neighbours to consider
#returns the accuracy of prediction of test data
def predictClassificationMinkowski(trainData, testData, k=1, n=1):
    good = 0
       
    for instance in testData:
        #get distances and idexes
        distances, indexes = calculateMinkowski(trainData[:,0:10], instance[0:10], n)
        # calculate top k indexes
        topk = indexes[0:k]
        #Calculate the max of the 1/distances^n to select the top class
        top = np.argmax(processClasses(trainData, topk, distances, n))
        # calculate correct predictions
        if top == int(instance[10]): good += 1
    # calculate accuracy
    acc = good / testData.shape[0]
    return acc


#method for finding k-NN 
#trainData - the train dataset including features and classes
#testData - the test dataset including features and classes
#k - the numer of nearest neighbours to consider
#returns the accuracy of prediction of test data
def predictClassification(trainData, testData, k=1, n=1):
    good = 0
        
    for instance in testData:
        #get distances and idexes
        distances, indexes = calculateDistances(trainData[:,0:10], instance[0:10])
        # calculate top k indexes
        topk = indexes[0:k]
        #Calculate the max of the 1/distances^n to select the top class
        top = np.argmax(processClasses(trainData, topk, distances, n))
        # calculate correct predictions
        if top == int(instance[10]): good += 1
    # calculate accuracy
    acc = good / testData.shape[0]
    return acc

a = np.array([[1.5,2.0,101.], [2.,5.,16.1], [3.,55.54,11]])
#scale(a)


train = r".\data\classification\trainingData.csv"
test = r".\data\classification\testData.csv"

#Load train data
trainData = loadData(train)

#Load test data
testData = loadData(test)

accuracy = predictClassification(trainData, testData, 10, 1)
accuracyMinkowski = predictClassificationMinkowski(trainData, testData, 10, 1)
print("Accuracy: {}".format(accuracy))

def report():

    accuracies = np.empty(shape=(19, 3))
    for i in range(10):
        accuracies[i] = [i+1, predictClassification(trainData, testData, 10, i+1), predictClassificationMinkowski(trainData, testData, 10, i+1)]

    np.savetxt("output3.csv", accuracies, delimiter=",")

