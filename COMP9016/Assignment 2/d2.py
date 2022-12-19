import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


cols = ['buying', 'maint', 'doors','persons','lug_boot','safety', 'classes']
file = 'C:\CIT MSc Repo\CIT MSc in AI\COMP9016\Assignment 2\Data\car.data'
data = pd.read_csv(file, sep=',', header=None, names=cols)
translations = {
    'buying': 
        {'vhigh': 4,
        'high': 3,
        'med': 2,
        'low': 1 },
    'maint':
        {'vhigh': 4,
        'high': 3,
        'med': 2,
        'low': 1 },
    'doors': 
        {'2': 1,
        '3': 2,
        '4': 3,
        '5more': 4},
    'persons': 
        {'2': 1,
        '4': 2,
        'more': 3},
    'lug_boot': 
        {'small': 1,
        'med': 2,
        'big': 3},
    'safety': 
        {'low': 1,
        'med': 2,
        'high': 3},
    'classes': 
        {'unacc': 1,
        'acc': 2,
        'good': 3,
        'vgood': 4}}

cd = data.copy()
for c, v in translations.items():
    for old, new in v.items():
        cd[c].mask(cd[c] == old, new, inplace=True)
    cd[c] = pd.to_numeric(cd[c])


totalRecords = cd.shape[0]
prior = {}
for c in cd.classes.unique():
    count = cd[cd['classes'] == c].shape[0] / totalRecords
    prior[c] = count
    print('Probability for class: {} is: {}'.format(c, count))

probabilityOfEvidence = {}
variableColumns = cols[:6]
for col in variableColumns:
    for val in cd[col].unique():
        key = '{} {}'.format(col, val)
        aaa = cd[cd[col] == val].shape[0]
        probabilityOfEvidence[key] = cd[cd[col] == val].shape[0] / totalRecords

print(probabilityOfEvidence)


#probabilityOfLikehood = pd.DataFrame(columns=['Evidence', '1', '2', '3', '4'])
probabilityOfLikehood = {}
for col in variableColumns:
    for val in cd[col].unique():
        
        key = '{} {}'.format(col, val)
        temp = {}
        for c in range(1,5):
            prob = cd[(cd[col] == val) & (cd['classes'] == c)].shape[0] / totalRecords
            temp[c] = prob
        probabilityOfLikehood[key] = temp

print(probabilityOfLikehood)


       













#fig, axs = plt.subplots(ncols=3, figsize=(18,3))
sns.countplot(x='class', data=data)
plt.show()