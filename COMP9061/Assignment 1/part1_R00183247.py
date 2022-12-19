
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

#method for finding k-NN 
#trainData - the train dataset including features and classes
#testData - the test dataset including features and classes
#k - the numer of nearest neighbours to consider
#returns the accuracy of prediction of test data
def predictClassification(trainData, testData, k=1):
    good = 0
    for instance in testData:
        #get distances and idexes
        distances, indexes = calculateDistances(trainData[:,0:10], instance[0:10])
        # calculate top k indexes
        topk = indexes[0:k]
        # select classes that correspond to the top k indexes
        selectedClasses = trainData[:,10][topk].astype(int)
        #This can be achieved as well using bincount and argmax
        top = int(Counter(selectedClasses).most_common(1)[0][0])
        # calculate correct predictions
        if top == int(instance[10]): good += 1
    # calculate accuracy
    acc = good / testData.shape[0]
    return acc



#Load train data
trainData = loadData(".\\data\\classification\\trainingData.csv")

#Load test data
testData = loadData(".\\data\\classification\\testData.csv")

accuracy = predictClassification(trainData, testData, 1)
print("Accuracy: {}".format(accuracy))

