from multiClassKernelPerceptron import MultiClassKernelPerceptron

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


if __name__ == '__main__':
    digitTrain = pd.read_csv("../dataset/mnist_train.csv") # type pandas.core.frame.DataFrame
    digitTest = pd.read_csv("../dataset/mnist_test.csv")

    # Loading training set and test set
    data = getDataset(digitTrain, digitTest)
    #print(data["imgTrain"])

    #print("# Iteration {0}".format(i))
    MCKernelPerceptron = MultiClassKernelPerceptron(epochNumber=5, polynomialDegree=3)

    # Training model
    print("Training Kernel Perceptron")
    MCKernelPerceptron.train(data["imgTrain"], data["labelTrain"])

    # Predicting with trained model
    print("Predicting Kernel Perceptron")
    yPred = MCKernelPerceptron.predict(data["imgTest"], data["labelTest"])

    print("Results")
    yTest = data["labelTest"].values
    accuracy = np.count_nonzero(yPred==yTest) / float(len(yTest))
    print(accuracy)

    df = pd.DataFrame({"x": data["labelTest"], "y": yPred})
    df_cond = df[df["x"] == df["y"]] # only correct predictions
    print(df_cond)
