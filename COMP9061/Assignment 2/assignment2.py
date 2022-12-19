import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV

from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection



def loadAndAnalyseAndPreProcessData():
    data = pd.read_csv(r'C:\CIT MSc Repo\CIT MSc in AI\COMP9061\Assignment 2\cardio_train.csv', sep=';')
    print(data.head())
    print(data.shape)
    print(data.isnull().values.any())
    # checking for imbalance
    cardio = data[data["cardio"]==1]
    noncardio = data[data["cardio"]==0]
    print(cardio.shape, noncardio.shape)

    # Checking the duplicates
    print("There is {} duplicated values in data frame".format(data.duplicated().sum()))

    # dropping the id from the data
    data.drop("id",axis=1,inplace=True)

    # converting the age from days to years
    data['age'] = (data['age'] / 365).round().astype('int')

    print(data.describe())


    return data

analysedData = loadAndAnalyseAndPreProcessData()

def outliersProcessding(data):
    

# Generate a boxplot to identify possible outliers
plt.boxplot(data['height'])
plt.show() 