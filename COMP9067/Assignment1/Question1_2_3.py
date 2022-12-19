import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt

'''
Method provided with the assignment.
I have changed the return type to return the tensors instead
'''
def loadData():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    # load the training and test data    
    (tr_x, tr_y), (te_x, te_y) = fashion_mnist.load_data()
    # reshape the feature data
    #print(tr_x.shape)
    tr_x = tr_x.reshape(tr_x.shape[0], 784)
    te_x = te_x.reshape(te_x.shape[0], 784)
    #print(tr_x.shape)
    # noramlise feature data
    tr_x = tr_x / 255.0
    te_x = te_x / 255.0
    print( "Shape of training features ", tr_x.shape)
    print( "Shape of test features ", te_x.shape)
    # one hot encode the training labels and get the transpose
    tr_y = np_utils.to_categorical(tr_y,10)
    tr_y = tr_y.T
    print ("Shape of training labels ", tr_y.shape)
    # one hot encode the test labels and get the transpose
    te_y = np_utils.to_categorical(te_y,10)
    te_y = te_y.T
    print ("Shape of testing labels ", te_y.shape)
    return  tf.Variable(tr_x, dtype=tf.float32),  \
            tf.Variable(tr_y,  dtype=tf.float32), \
            tf.Variable(te_x,  dtype=tf.float32), \
            tf.Variable(te_y,  dtype=tf.float32)


@tf.function
def forward_pass(x, w1, b1, w2, b2, w3, b3):
    #ReLu Layer 1 (Hidden Layer)
    t1 = tf.add(tf.matmul(x, w1), b1)
    relu1 = tf.math.maximum(t1, 0)
    #ReLu Layer2 (Hidden Layer)
    t2 = tf.add(tf.matmul(relu1, w2), b2)
    relu2 = tf.math.maximum(t2, 0)
    #Softmax Layer 
    t3 = tf.exp(tf.add(tf.matmul(relu2, w3), b3))
    sm =  t3 / tf.reduce_sum(t3, axis= 1, keepdims=True)
    return sm


@tf.function
def cross_entropy(predicted, y, w1, w2, w3, regRate, reg="None"):
    #y_pred = tf.transpose(predicted)
    ce = -tf.reduce_sum(tf.transpose(y) * tf.math.log(predicted), axis=1)
    #loss = tf.reduce_mean(ce) + tf.multiply(regRate, weights)
    if reg == "L1":
        weights = tf.reduce_sum(tf.math.abs(w1)) + tf.reduce_sum(tf.math.abs(w2))+ tf.reduce_sum(tf.math.abs(w3))
        loss = tf.reduce_mean(ce) + tf.multiply(regRate, weights)
    elif reg == "L2":
        weights = tf.reduce_sum(tf.math.square(w1)) + tf.reduce_sum(tf.math.square(w2))+ tf.reduce_sum(tf.math.square(w3))
        loss = tf.reduce_mean(ce) + tf.multiply(regRate, weights)
    else:
        loss = tf.reduce_mean(ce)
    return loss

#https://www.tensorflow.org/guide/keras/train_and_evaluate
@tf.function
def calculate_accuracy(predicted, y):
    #Either transpose or run argmax on axis 1
    predicted = tf.argmax(predicted, axis=1)
    y = tf.argmax(y, axis=0)
    values = tf.cast(y, tf.int32) == tf.cast(predicted, tf.int32)
    values = tf.cast(values, tf.float32)
    accuracy = tf.reduce_mean(values)
    return accuracy

@tf.function
def calculate_accuracy2(predicted, y):
    predictions = tf.round(tf.transpose(predicted))
    print(predictions)
    print(y)
    correctPredictions = tf.cast(tf.equal(predictions, y), dtype=tf.float32)
    #print(correctPredictions)
    accuracy = tf.reduce_mean(correctPredictions)
    return accuracy


def run(iterations, optimiser, numberOfClasses, dataSize, regRate=0, reg="None"):
    w1 = tf.Variable(tf.random.normal([dataSize, 300], mean=0.0, stddev=0.05, dtype=tf.float32))
    b1 = tf.Variable(tf.zeros([1, 300]))
    w2 = tf.Variable(tf.random.normal([300, 100], mean=0.0, stddev=0.05, dtype=tf.float32))
    b2 = tf.Variable(tf.zeros([1, 100]))
    w3 = tf.Variable(tf.random.normal([100, numberOfClasses], mean=0.0, stddev=0.05, dtype=tf.float32))
    b3 = tf.Variable(tf.zeros([1, numberOfClasses]))
    for i in range(iterations):
        with tf.GradientTape() as tape:

            predictions = forward_pass(tr_x, w1, b1, w2, b2, w3, b3)
            loss = cross_entropy(predictions, tr_y, w1, w2, w3, regRate, reg )
            testPredictions = forward_pass(te_x, w1, b1, w2, b2, w3, b3)
            history['trainAccuracy'].append(calculate_accuracy(predictions, tr_y).numpy())
            history['trainLoss'].append(loss.numpy())
            history["testAccuracy"].append(calculate_accuracy(testPredictions, te_y).numpy())
            history["testLoss"].append(cross_entropy(testPredictions, te_y, w1, w2, w3, regRate, reg).numpy())

        gradients = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        optimiser.apply_gradients(zip(gradients, [ w1, b1, w2, b2, w3, b3]))
    
    
    #
    # Print final training accuracy and regularized loss
    #
    

    #
    # Calculate predictions, loss and accuracy for test set
    #
    #y_pred = forward_pass(te_x, w, b)
    #accuracy = calculate_accuracy(y_pred, te_y)
    #loss = cross_entropy(y_pred, te_y)
    
    #
    # Print test accuracy and regularized loss
    #
    print('Train accuracy:', history['trainAccuracy'][-1])
    print('Test accuracy:', history['testAccuracy'][-1])
    print('Train loss:', history['trainLoss'][-1])
    print('Test loss:', history['testLoss'][-1], '\n')

       # print("Iteration",i,":Test Loss=",CT.numpy(), "Test Acc:",accuracyTest.numpy())
        #print(accuracy)
    
    plt.plot(history['trainLoss'])
    plt.plot(history['trainAccuracy'])
    plt.plot(history['testLoss'])
    plt.plot(history['testAccuracy'])
    plt.xlabel('Epoch #')
    plt.ylabel('Loss / Accuracy')
    plt.title("Training Loss and Accuracy")
    
    plt.show()

numberOfClasses = 10
accuracy = {}
regRate = tf.constant(0.001, dtype=tf.float32)
iterations = 100
history ={}
history["trainAccuracy"] = []
history['trainLoss'] = []
history["testAccuracy"] = []
history['testLoss'] = []
tr_x, tr_y, te_x, te_y = loadData()
inputSize = tr_x.shape[1]
run(iterations, tf.keras.optimizers.Adam(), numberOfClasses, tr_x.shape[1], regRate, "L2")