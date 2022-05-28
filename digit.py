from multiClassKernelPerceptron import MultiClassKernelPerceptron

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getDataset(data1, data2):
    # training data
    labels1 = data1.label
    digits1 = data1.drop(['label'], axis=1)

    # test data
    labels2 = data2.label
    digits2 = data2.drop(['label'], axis=1)

    size = 10000
    sizeTest = 10000

    return {"imgTrain": digits1[:size],
            "imgTest": digits2[:sizeTest],
            "labelTrain": labels1[:size],
            "labelTest": labels2[:sizeTest]
    }

def plotStats(x):
    plt.figure(figsize=(20, len(x)))
    plt.subplot(1, 2, 1)
    for poly in range(V1.shape[0]):
        plt.plot(x, V1[poly], label='polynomial degree = {0}'.format(poly+1))
    plt.xlabel('Number of epochs')
    plt.ylabel('Error values')
    plt.legend(loc='upper left')
    plt.title('Perceptrons average')

    plt.subplot(1, 2, 2)
    for poly in range(V2.shape[0]):
        plt.plot(x, V2[poly], label='polynomial degree = {0}'.format(poly+1))
    plt.xlabel('Number of epochs')
    plt.ylabel('Error values')
    plt.legend(loc='upper left')
    plt.title('Perceptron achieving smallest training error')

    return plt.show()


if __name__ == '__main__':
    digitTrain = pd.read_csv("../input/trainPCA.csv", index_col=0)
    digitTest = pd.read_csv("../input/testPCA.csv", index_col=0)

    ## Loading training set and test set
    data = getDataset(digitTrain, digitTest)

    epochNumber = 20
    polynomialDegree = 8

    V1 = np.zeros((polynomialDegree,int(epochNumber/5)))
    V2 = np.zeros((polynomialDegree,int(epochNumber/5)))
    accuracyV1 = 0
    accuracyV2 = 0

    #occ = []
    #occurrences = [np.count_nonzero(data["labelTest"] == i) for i in range(10)]
    #print(occurrences)

    for degree in range(polynomialDegree): #polynomialDegree
        print("# Polynomial Degree: {0}".format(degree+1))
        MCKernelPerceptron = MultiClassKernelPerceptron(epochNumber, degree+1)

        # Training model
        print("Training Kernel Perceptron")
        MCKernelPerceptron.train(data["imgTrain"], data["labelTrain"])

        # Predicting with trained model
        print("Predicting Kernel Perceptron")
        yPred = MCKernelPerceptron.predict(data["imgTest"], data["labelTest"])
        #print(yPred)

        accV1 = 0
        accV2 = 0

        n,m = yPred.shape
        c1 = 0
        c2 = 0

        for i in range(n):
            df = pd.DataFrame({"label": data["labelTest"], "pred label": yPred[i]})
            correct = df[df["label"] == df["pred label"]]
            accuracy = float(correct.shape[0] / m) # misclassified predictions
            error = 1 - accuracy
            # even indexes represent interim results of Perceptron version 1
            if i%2 == 0:
                V1[degree,c1] = error
                accV1 += accuracy
                c1 += 1
            else: # odd indexes represent interim results of Perceptron version 2
                V2[degree,c2] = error
                accV2 += accuracy
                c2 += 1

            occ.append([np.count_nonzero(yPred[i] == j) for j in range(10)])

        # calculating the accuracy of each version for every polynomial degree
        accuracyV1 += accV1/int(n/2)
        accuracyV2 += accV2/int(n/2)

    #out = np.zeros(10)
    #for i in range(len(occ)):
    #    for j in range(10):
    #        out[j] += occ[i][j]
    #out = out/len(occ)
    #print(''.join("{:.2f} ".format(x) for x in out))

    print("Accuracy in version1 (predictors average): {:.2f}".format(accuracyV1/polynomialDegree))
    print("Accuracy in version2 (predictor achieving the smallest training error): {:.2f}".format(accuracyV2/polynomialDegree))

    xAxis = [i+5 for i in range(0,epochNumber,5)]
    plotStats(xAxis)
