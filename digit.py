from multiClassKernelPerceptron import MultiClassKernelPerceptron
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np # mathematical functions
import pandas as pd # data analysis - CSV file I/O ??
import matplotlib.pyplot as plt # like MATLAB
import sys

def getDataset(data1, data2):
    # training data
    size = 1000
    data1 = data1.sample(frac=1).reset_index(drop=True) # shuffling
    labels1 = data1.label # <class 'pandas.core.series.Series'>
    digits1 = data1.drop(['label'], axis=1) # <class 'pandas.core.frame.DataFrame'>

    # test data
    sizeTest = int(size/2)
    labels2 = data2.label
    digits2 = data2.drop(['label'], axis=1)


    return {"imgTrain": digits1[:size],
            "imgTest": digits2[:sizeTest],
            "labelTrain": labels1[:size],
            "labelTest": labels2[:sizeTest]
    }

def getDatasetPCA(data1, data2):
    # training data
    #size = 1000
    data1 = data1.sample(frac=1).reset_index(drop=True) # shuffling
    labels1 = data1.label # <class 'pandas.core.series.Series'>
    digits1 = data1.drop(['label'], axis=1) # <class 'pandas.core.frame.DataFrame'>

    # test data
    #sizeTest = int(size/2)
    labels2 = data2.label
    digits2 = data2.drop(['label'], axis=1)

    size1,feat1 = digits1.shape
    print(feat1)


    ## Preprocessing data
    scaler = StandardScaler()

    # Fit on training set only
    scaler.fit(digits1)
    # Apply transform to both the training set and the test set
    digits1 = scaler.transform(digits1)
    digits2 = scaler.transform(digits2)

    # Make an instance of the Model
    pca = PCA(.95)

    pca.fit(digits1)

    digits1 = pca.transform(digits1)
    digits2 = pca.transform(digits2)

    size11,feat11 = digits1.shape
    print(feat11)


    return {"imgTrain": digits1,
            "imgTest": digits2,
            "labelTrain": labels1,
            "labelTest": labels2
    }


if __name__ == '__main__':
    #print(np.zeros((2,3,5,10)))
    digitTrain = pd.read_csv("../dataset/mnist_train.csv") # type pandas.core.frame.DataFrame
    digitTest = pd.read_csv("../dataset/mnist_test.csv")

    ## Loading training set and test set
    data = getDataset(digitTrain, digitTest)

    epochNumber = 20
    polynomialDegree = 3

    #print("# Iteration {0}".format(i))
    MCKernelPerceptron = MultiClassKernelPerceptron(epochNumber, polynomialDegree)

    # Training model
    print("Training Kernel Perceptron")
    MCKernelPerceptron.train(data["imgTrain"], data["labelTrain"])

    # Predicting with trained model
    print("Predicting Kernel Perceptron")
    yPred = MCKernelPerceptron.predict(data["imgTest"], data["labelTest"])
    print(yPred.shape)
    print(yPred)

    print("Results")
    yTest = data["labelTest"].values
    #accuracy = float(np.count_nonzero(yPred==yTest) / len(yTest))
    #print("{:.2f}".format(accuracy))

    df = pd.DataFrame({"x": data["labelTest"], "y1": yPred[0], "y2": yPred[1], "y3": yPred[2], "y4": yPred[3], "y5": yPred[4], "y6": yPred[5], "y7": yPred[6], "y8": yPred[7]})
    df_cond1 = df[df["x"] == df["y1"]] # only correct predictions
    print(df_cond1)
    df_cond2 = df[df["x"] == df["y2"]]
    print(df_cond2)
    df_cond3 = df[df["x"] == df["y3"]]
    print(df_cond3)
    df_cond4 = df[df["x"] == df["y4"]]
    print(df_cond4)
    df_cond5 = df[df["x"] == df["y5"]]
    print(df_cond5)
    df_cond6 = df[df["x"] == df["y6"]]
    print(df_cond6)
    df_cond7 = df[df["x"] == df["y7"]]
    print(df_cond7)
    df_cond8 = df[df["x"] == df["y8"]]
    print(df_cond8)


    #"{:.2f}".format(
