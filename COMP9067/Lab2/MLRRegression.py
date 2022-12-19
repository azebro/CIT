
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# This function will take in all the feature data X
# as well as the current coefficient and bias values
# It should multiply all the feature value by their associated 
# coefficient and add the bias. It should then return the predicted 
# y values
def hypothesis(X, coefficients, bias):
    
    # TODO: Calculate and return predicted results
    
    return 0




def calculateRSquared(bias, coefficients,X, Y):
    
    predictedY = hypothesis(X, coefficients, bias)
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    
    sumSqRes = np.sum((predictedY - Y)**2)
    
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2
    


        
    

def gradient_descent(bias, coefficients, alpha, X, Y, max_iter):


    length = len(X)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    X1 = X.transpose()
    for num in range(0, max_iter):
        A = np.dot(coefficients.transpose(), X1) + bias
        errorValues = A - Y
        db = np.sum(errorValues)/2*length
        d_alpha = np.dot(X1, errorValues)/ 2*length
        coefficients = coefficients - alpha * d_alpha
        bias = bias - alpha * db
        print(errorValues)
        
    
        

        
        # TODO: 
        # Calculate predicted y values for current coefficient and bias values 
        # calculate and update bias using gradient descent rule
        # Update each coefficient value in turn using gradient descent rule

    
    # calculate R squared value for current coefficient and bias values
    rSquared = calculateRSquared(bias, coefficients,X, Y)
    print ("Final R2 value is ", rSquared)
    plt.plot(errorValues)
    plt.show()

    # plot the cost for each iteration of gradient descent
    
    return bias, coefficients


def multipleLinearRegression(X, Y):

    # set the number of coefficients equal to the number of features
    coefficients = np.zeros(X.shape[1])
    bias = 0.0
    
    alpha = 0.01 # learning rate
    
    max_iter=100

    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X, Y, max_iter)
    
    
    
    
def main():
    
    df = pd.read_csv(r"C:\GitHub\CIT\COMP9067\Lab2\boston.csv")
    df = df.dropna()
    print (df.shape)
    
    data = df.values

     
    # Seperate teh features from the target feature    
    Y = data[:, -1]
    X = data[:, :-1]
    
    # Standardize each of the features in the dataset. 
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
     
    # run regression function
    multipleLinearRegression(X, Y)
    

    

main()
