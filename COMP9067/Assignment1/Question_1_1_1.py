import tensorflow as tf
from keras.utils import np_utils

import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

# load the training and test data    
(tr_x, tr_y), (te_x, te_y) = fashion_mnist.load_data()

# reshape the feature data
print(tr_x.shape)

tr_x = tr_x.reshape(tr_x.shape[0], 784)
te_x = te_x.reshape(te_x.shape[0], 784)
print(tr_x.shape)

# noramlise feature data
train_x = tr_x / 255.0
test_x = te_x / 255.0

print( "Shape of training features ", tr_x.shape)
print( "Shape of test features ", te_x.shape)


# one hot encode the training labels and get the transpose
tr_y = np_utils.to_categorical(tr_y,10)
train_y = tr_y.T
print ("Shape of training labels ", tr_y.shape)

# one hot encode the test labels and get the transpose
te_y = np_utils.to_categorical(te_y,10)
test_y = te_y.T
print ("Shape of testing labels ", te_y.shape)

#tr_xT = tr_x.T
#print(tr_xT.shape)
input_size = tr_x.shape[1]
num_classes = 10
weights = tf.Variable(tf.random.normal([input_size, num_classes]))
bias = tf.Variable(tf.zeros([1, num_classes]))


@tf.function
def forward_pass(x, w, b):
    # Softmax layer
    logits = tf.add(tf.matmul(tf.cast(x, dtype=tf.float32), tf.cast(w, dtype=tf.float32)), b)
    softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis=-1, keepdims=True)
    return softmax

@tf.function
def cross_entropy(y_pred, y):
    # Transpose y_pred to allow multiplication
    # Calculate Cross-Entropy Loss per instance
    # Calculate total loss
    y_pred = tf.transpose(y_pred)
    ce = -tf.reduce_sum(y * tf.math.log(y_pred), axis=0)
    loss = tf.reduce_mean(ce)
    return loss

@tf.function
def calculate_accuracy(y_pred, y):
    # Take the transpose of y_pred to bring y_pred and y into same shape
    # Create vector with 1.0 if prediction and y have highest probability for same class, 0.0 otherwise
    # Caluculate average of that vector to get accuracy
    predictions = tf.transpose(y_pred)
    predictions_correct = tf.cast(tf.equal(tf.argmax(predictions, axis=0), tf.argmax(y, axis=0)), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    return accuracy
history ={}
history['acc'] = []
history['loss'] = []

def optimize(epochs, optimizer):
    input_size = train_x.shape[1]
    num_classes = 10

    #
    # Initialize weights and biases for Softmax layer
    #
    w = tf.Variable(tf.random.normal([input_size, num_classes], mean=0.0, stddev=0.05))
    b = tf.Variable(tf.zeros([1, num_classes]))

    #
    # Iterate through epochs
    #
    for i in range(epochs):
        with tf.GradientTape() as tape:
            #
            # Execute 1 training cycle
            #
            y_pred = forward_pass(train_x, w, b)
            loss = cross_entropy(y_pred, train_y)

            #
            # Add epoch accuracy and loss to history
            #
            history['acc'].append(calculate_accuracy(y_pred, train_y))
            history['loss'].append(loss)
        
        #
        # Caluclate the gradients for weights and biases 
        # Update weights and biases
        #    
        gradients = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradients, [w, b]))

    #
    # Print final training accuracy and regularized loss
    #
    print('The train accuracy is:', history['acc'][-1].numpy())
    print('The train loss     is:', history['loss'][-1].numpy())

    #
    # Calculate predictions, loss and accuracy for test set
    #
    y_pred = forward_pass(test_x, w, b)
    accuracy = calculate_accuracy(y_pred, test_y)
    loss = cross_entropy(y_pred, test_y)
    
    #
    # Print test accuracy and regularized loss
    #
    print('The test accuracy is:', accuracy.numpy())
    print('The test loss     is:', loss.numpy(), '\n')



optimize(100, tf.keras.optimizers.Adam())