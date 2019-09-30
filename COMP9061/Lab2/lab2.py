

import os
from math import log
from collections import Counter


'''
Load data and count the occurances.
TODO: look into data pre-processing
'''
def countOccurances(folder):
    allWords = []
    files = os.listdir(folder)
    for file in files:
        f = open(folder + file, 'r', encoding='utf8')
        allWords += f.read().lower().split()
        f.close()
    return Counter(allWords), len(allWords), len(files)


def countPw(vocabulary, totalNumberOfWords):
    V = sum(vocabulary.values())
    pw  = {}
    for item in vocabulary:
        pw[item] =  (vocabulary[item] + 1) / (totalNumberOfWords + V)
    return pw

'''
TODO: THat requires re-thinking and optimisation
'''
def processMissing(positive, negative):
    tempPos = set(negative) - set(positive)
    tempNeg = set(positive) - set(negative)
    outputPos = {}
    outputNeg = {}
    for item in tempPos:
        outputPos[item] = 0
    for item in tempNeg:
        outputNeg[item] = 0
    return outputPos, outputNeg

'''
Train the model.
TODO: look at optimising the missing words in neg/pos
'''
def trainDataset(pathToPositive, pathToNegative):
    countsNegative, allNegative, instancesNegative = countOccurances(pathNegative)
    countsPositive , allPositive, instancesPositive = countOccurances(pathPositive)
    missingPos, missingNeg = processMissing(countsPositive, countsNegative)
    countsNegative.update(missingNeg)
    countsPositive.update(missingPos)
    pWpos = countPw(countsPositive, allPositive)
    pWneg = countPw(countsNegative, allNegative)
    cPos = instancesPositive / (instancesPositive + instancesNegative)
    cNeg = instancesNegative / (instancesNegative + instancesPositive)
    return pWpos, pWneg, cPos, cNeg


def calculatePwC(data, pWs):
    probabilitySum = 0
    for k, v in data.items():
        if k in pWs:
            probabilitySum += log((v * pWs[k]))
    return probabilitySum

def classify(data, cP, cN, pWpos, pWneg):
    pPosData = log(cP) + calculatePwC(data, pWpos)
    pNegData = log(cN) + calculatePwC(data, pWneg)
    if pPosData > pNegData:
        return True
    return False

def test(folder, c):
    files = os.listdir(folder)
    outcomes = []
    for file in files:
        f = open(folder + file, 'r', encoding='utf8')
        allWords = f.read().lower().split()
        f.close()
        outcome = classify(Counter(allWords), cP, cN, pWpos, pWneg)
        outcomes.append(outcome)
        #print("Classification for file:" + file + " : " + str(outcome ))
    accuracy = len([o for o in outcomes if o == c]) / len(files)
    print("Accuracy for: " + str(c) + " is " + str(accuracy))
    return

pathNegative = 'C:\\Users\\adamze\\Downloads\\data\\train\\neg\\'
pathPositive = 'C:\\Users\\adamze\\Downloads\\data\\train\\pos\\'

pWpos, pWneg, cP, cN = trainDataset(pathPositive, pathNegative)


test('C:\\Users\\adamze\\Downloads\\data\\test\\neg\\', False)
test('C:\\Users\\adamze\\Downloads\\data\\test\\pos\\', True)


#print(pWpos)
#print(pWneg)

#uniqueWords = set(allWords)


#print(counts)
#print(countsNegative)
#print(countsPositive)
#print(pWpos)
#print(pWneg)






