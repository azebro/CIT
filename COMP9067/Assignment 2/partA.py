import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd


def loadDataH5():
    with h5py.File(r'C:\CIT MSc Repo\CIT MSc in AI\COMP9067\Assignment 2\data\data1.h5','r') as hf: 
        trainX = np.array(hf.get('trainX')) 
        trainY = np.array(hf.get('trainY')) 
        valX = np.array(hf.get('valX')) 
        valY = np.array(hf.get('valY')) 
        print (trainX.shape,trainY.shape) 
        print (valX.shape,valY.shape)
    return trainX, trainY, valX, valY




def singleLayers(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = keras.models.Sequential()
    inputShape = (height, width, depth)
    model.add(keras.layers.Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Dense(classes, activation='softmax'))
    return model

def deeper1(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = keras.models.Sequential()
    inputShape = (height, width, depth)
    model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000,activation='relu'))
    model.add(keras.layers.Dense(classes, activation='softmax'))
    return model

def deeper2(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = keras.models.Sequential()
    inputShape = (height, width, depth)
    model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000,activation='relu'))
    model.add(keras.layers.Dense(classes, activation='softmax'))
    return model

def deeper3(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = keras.models.Sequential()
    inputShape = (height, width, depth)
    model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape, activation='relu'))
        #model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(256, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(256, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(512, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Conv2D(256, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(keras.layers.Conv2D(1024, (1, 1), padding="same", activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
   

    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000,activation='relu'))
    model.add(keras.layers.Dense(classes, activation='softmax'))
    return model

def executeModel(model):
    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    h = model.fit(trainX, trainY, batch_size=16, epochs=NUM_EPOCHS, validation_split=0.1)
    results = model.evaluate(testX, testY)
    return h, results


trainX, trainY, testX, testY = loadDataH5()
NUM_EPOCHS = 50
opt = keras.optimizers.SGD(lr=0.01)
model = deeper1(width=128, height=128, depth=3, classes=17)
print(model.summary())
H, results = executeModel(model, 8)
trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
      shear_range=0.2,
      zoom_range=0.2,
      rotation_range=30,
      horizontal_flip=True,
      featurewise_center=True,
      width_shift_range=0.2,
      height_shift_range=0.2)
train_generator = trainDataGenerator.flow(trainX, trainY, batch_size=32)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
'''

H = model.fit(train_generator, 
                        validation_data=(testX, testY), 
                        steps_per_epoch=len(trainX)/ 32, 
                        epochs = NUM_EPOCHS)









#H = model.fit(trainX, trainY, batch_size=16, epochs=NUM_EPOCHS, validation_split=0.1)
'''
models = [model1(), model2(), model3(), model4(), model5(), model6(), model7(), model8(), model9(), model10()]
ensambleData = []

def loadModels(models):
    loadedModels = []
    for model in models:
        name = model.__name__
        path = format("output/{name}_weights.hdf5")
        temp = keras.load_model(path)
        temp.name = name
        loadedModels.append(temp)
    return loadedModels




print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()