import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt



def loadData():
    with h5py.File(r'C:\CIT MSc Repo\CIT MSc in AI\COMP9067\Assignment1\data\data.h5','r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        allTrain = hf.get('trainData')
        allTest = hf.get('testData')
        npTrain = np.array(allTrain)
        npTest = np.array(allTest)
    print('Shape of the array dataset_1: \n', npTrain.shape)
    print('Shape of the array dataset_2: \n', npTest.shape)
    return npTrain[:,:-1], npTrain[:, -1], npTest[:,:-1], npTest[:, -1]
trainX, trainY, testX, testY = loadData()
outputs = []

def initialTask():
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=5, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    print(results)

def twoLayers():
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(200, activation=tf.nn.relu))
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    print(results)

def threeLayers():
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(400, activation=tf.nn.relu))
    model.add(layers.Dense(200, activation=tf.nn.relu))
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    print(results)

def fourLayers():
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(600, activation=tf.nn.relu))
    model.add(layers.Dense(400, activation=tf.nn.relu))
    model.add(layers.Dense(200, activation=tf.nn.relu))
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    print(results)

def fiveLayers():
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(800, activation=tf.nn.relu))
    model.add(layers.Dense(600, activation=tf.nn.relu))
    model.add(layers.Dense(400, activation=tf.nn.relu))
    model.add(layers.Dense(200, activation=tf.nn.relu))
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    print(results)


def fourLayersDropout():
    model = tf.keras.models.Sequential()
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(400, activation=tf.nn.relu))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(200, activation=tf.nn.relu))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    print(results)



def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 4)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
def regulariseFour():
    model = tf.keras.models.Sequential()
    model.add(layers.Flatten(input_shape=(784,)))
    model.add(layers.Dense(600, activation=tf.nn.relu))
    model.add(layers.Dense(400, activation=tf.nn.relu))
    model.add(layers.Dense(200, activation=tf.nn.relu))
    model.add(layers.Dense(10, activation= tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=5, validation_split=0.1)
    #results = model.evaluate(testX, testY)
    #outputs.append(results)
    #print(results)

def buildModel(layersList):
    model = tf.keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(784,)))
    for layer in layersList:
        model.add(layers.Dense(layer["size"], activation=layer["activation"]))
        if layer["applyDropout"]:
            model.add(layers.Dropout(layer["dropRate"]))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
    

def executeModel(hidden):
    layers = []
    for layer in hidden:
        layers.append({"size": layer[0], "activation": tf.nn.relu, "applyDropout": layer[1], "dropRate": layer[2]})
    layers.append({"size": 10, "activation": tf.nn.softmax, "applyDropout": False, "dropRate": 0.3})
    model = buildModel(layers)
    history = model.fit(trainX, trainY, epochs=2, batch_size=256)
    results = model.evaluate(testX, testY)
    outputs.append(results)
    return history, results

#initialTask()

def executeModels():
    tasks = []
    labels = []
    losses = []
    accuracies = []
    tasks.append([[1000, False, 0], [800, False, 0], [600, False, 0], [400, False, 0], [200, False, 0]])
    tasks.append([[800, False, 0], [600, False, 0], [400, False, 0], [200, False, 0]])
    tasks.append([[600, False, 0], [400, False, 0], [200, False, 0]])
    tasks.append([[400, False, 0], [200, False, 0]])
    tasks.append([[200, False, 0]])
    for task in tasks:
        h, r = executeModel(task)
        networkLength = len(task) + 1
        print("Executed model with {} layers. \n".format( networkLength ))
        labels.append(str(networkLength) + " Layers")
        losses.append(round(r[0], 4))
        accuracies.append(round(r[1], 4))
         


    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    width = 0.35
    rects1 = ax.bar(x - width/2, losses, width, label='Loss')
    rects2 = ax.bar(x + width/2, accuracies, width, label='Accuracy')
    ax.set_ylabel('Loss / Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    plt.show()
    


    
executeModels()

print("Initial")
initialTask()
print("Two layers")
twoLayers()
print("Three layers")
threeLayers()
print("Four layers")
fourLayers()
print("Five layers")
fiveLayers()
#print("Regularised")
#regulariseFour()

print("Dropout")
fourLayersDropout()

fig, ax = plt.subplots()
labels = ["Reference",  "2 Layers", "3 Layers", "4 Layers", "5 Layers", "Drop"]
x = np.arange(len(labels))
width = 0.35
results = np.array(outputs)
loss = results[:, 0].reshape(1,6)[0]
rects1 = ax.bar(x - width/2, loss, width, label='Loss')
rects2 = ax.bar(x + width/2, results[:, 1], width, label='Accuracy')
ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()