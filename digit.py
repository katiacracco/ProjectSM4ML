from multiClassKernelPerceptron import MultiClassKernelPerceptron

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sys

def getDataset(data1, data2):
    # training data
    labels1 = data1.label
    digits1 = data1.drop(['label'], axis=1)

    # test data
    labels2 = data2.label
    digits2 = data2.drop(['label'], axis=1)

    size = 1000

    return {"imgTrain": digits1[:size],
            "imgTest": digits2,
            "labelTrain": labels1[:size],
            "labelTest": labels2
    }

def plotStats(x):

    #print(xAxis.shape)
    #print(V1[0].shape)
    #print(V1.shape)
    plt.figure(figsize=(20, len(x)))
    plt.subplot(1, 2, 1)
    for poly in range(V1.shape[0]):
        plt.plot(x, V1[poly], label='polynomial degree = {0}'.format(poly+1))
    #plt.plot(x, V1[1], label='polynomial degree = 2')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error values')
    plt.legend(loc='lower right')
    plt.title('Perceptrons average')

    plt.subplot(1, 2, 2)
    for poly in range(V2.shape[0]):
        plt.plot(x, V2[poly], label='polynomial degree = {0}'.format(poly+1))
    #plt.plot(x, V2[1], label='polynomial degree = 2')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error values')
    plt.legend(loc='lower right')
    plt.title('Perceptron achieving smallest training error')

    return plt.show()


if __name__ == '__main__':
    digitTrain = pd.read_csv("trainPCA.csv", index_col=0) # type pandas.core.frame.DataFrame
    digitTest = pd.read_csv("testPCA.csv", index_col=0)

    ## Loading training set and test set
    data = getDataset(digitTrain, digitTest)

    epochNumber = 30
    polynomialDegree = 8

    V1 = np.zeros((polynomialDegree,int(epochNumber/5)))
    V2 = np.zeros((polynomialDegree,int(epochNumber/5)))


    for degree in range(polynomialDegree): #polynomialDegree
        print("# Polynomial Degree: {0}".format(degree+1))
        MCKernelPerceptron = MultiClassKernelPerceptron(epochNumber, degree+1)

        #occurrences = [np.count_nonzero(data["labelTrain"] == i) for i in range(10)]
        #print(occurrences)

        # Training model
        print("Training Kernel Perceptron")
        MCKernelPerceptron.train(data["imgTrain"], data["labelTrain"])

        # Predicting with trained model
        print("Predicting Kernel Perceptron")
        yPred = MCKernelPerceptron.predict(data["imgTest"], data["labelTest"])
        #print(yPred.shape)
        #print(yPred)

        #occurrences = [np.count_nonzero(data["labelTest"] == i) for i in range(10)]
        #print(occurrences)

        #df = pd.DataFrame({"x": data["labelTest"], "y1": yPred[0], "y2": yPred[1], "y3": yPred[2], "y4": yPred[3], "y5": yPred[4], "y6": yPred[5], "y7": yPred[6], "y8": yPred[7]})

        accV1 = 0
        accV2 = 0

        n,m = yPred.shape
        #print(n) #8
        #print(m) #500
        c1 = 0
        c2 = 0

        for i in range(n):
            df = pd.DataFrame({"label": data["labelTest"], "pred label": yPred[i]})
            correct = df[df["label"] == df["pred label"]] # epoch 5 - v1
            accuracy = float(correct.shape[0] / m) # misclassified predictions
            error = 1 - accuracy
            # first half are interim results of Perceptron version 1
            if i%2 == 0:
                V1[degree,c1] = error
                accV1 += accuracy
                c1 += 1
            else: # second half are interim results of Perceptron version 2
                V2[degree,c2] = error
                accV2 += accuracy
                c2 += 1
            #occ = [np.count_nonzero(yPred[i] == lab) for lab in range(10)]
            #print(occ)

        print("Accuracy in version1 (predictors average): {:.2f}".format(accV1/int(n/2)))
        print("Accuracy in version2 (predictor achieving the smallest training error): {:.2f}".format(accV2/int(n/2)))

    xAxis = [i+5 for i in range(0,epochNumber,5)]
    plotStats(xAxis)
