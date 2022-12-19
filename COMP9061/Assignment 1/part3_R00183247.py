
import numpy as np


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


def processClasses(trainData, distances, n):
    #applying the formula of f(x) = (sum(f(xi) / d^n) / sum(1 / d^n)). 
    #simplyfying the formula as 1/d^n * f(x) = f(x)/d^n for the top part
    regValues = np.divide(np.sum(np.divide(trainData, np.power(distances, 2))),np.sum(np.divide(1, np.power(distances, 2))))
    return regValues

def calculateR2(testData, predictedData):
    #calculate the average of test data
    averageTest = np.mean(testData)
    #calculate the total sum of squares
    sumOfSquares = np.sum(np.power((testData - averageTest),2))
    #calculate sum of residuals
    sumOfResiduals = np.sum(np.power((predictedData - testData),2))
    #calculate the R2 as 1 - sumOfResiduals / sumOfSquares
    r2 = 1 - np.divide(sumOfResiduals, sumOfSquares)
    return r2

    

#method for finding k-NN 
#trainData - the train dataset including features and classes
#testData - the test dataset including features and classes
#k - the numer of nearest neighbours to consider
#returns the accuracy of prediction of test data
def predictClassification(trainData, testData, k=10, n=2):
    good = 0
    outputs = []
    for instance in testData:
        #get distances and idexes
        distances, indexes = calculateDistances(trainData[:,:12], instance[:12])
        # calculate top k indexes
        topk = indexes[0:k]
        #Calculate distance weighted regression prediction
        top = processClasses(trainData[topk][:,12], distances[topk], n)
        # calculate correct predictions
        outputs.append(top)
    # calculate accuracy using r^2 formula
    r2 = calculateR2(testData[:,12], outputs)
    return r2

train = r".\data\regression\trainingData.csv"
test = r".\data\regression\testData.csv"

#Load train data
trainData = loadData(train)

#Load test data
testData = loadData(test)
#invoke the k-NN prediction function supplying train data, test data, k and n
accuracy = predictClassification(trainData, testData, 3, 2)

print("Accuracy: {0:.4f}".format(accuracy))

