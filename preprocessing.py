from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

# preprocessing data
def applyPCA(data1, data2):
    # training data
    data1 = data1.sample(frac=1).reset_index(drop=True) # shuffling
    labels1 = data1.label
    digits1 = data1.drop(['label'], axis=1)
    # test data
    labels2 = data2.label
    digits2 = data2.drop(['label'], axis=1)

    # preprocessing data
    digits1 = digits1/255.0
    digits2 = digits2/255.0
    # make an instance of PCA model
    pca = PCA(0.9)
    # fit on training set only
    pca.fit(digits1)
    # apply transform to both training and test set
    digits1pca = pca.transform(digits1)
    digits2pca = pca.transform(digits2)

    size = 10000

    # creating a file with PCA training set
    df1 = pd.DataFrame(digits1pca[:size])
    df1['label'] = labels1[:size]
    df1.to_csv('trainPCA.csv')

    # creating a file with PCA test set
    df2 = pd.DataFrame(digits2pca)
    df2['label'] = labels2
    df1.to_csv('testPCA.csv')

    return {"imgTrain": digits1pca[:size],
        "imgTest": digits2pca,
        "labelTrain": labels1[:size],
        "labelTest": labels2
    }

# computing polynomial kernel
def polyKernel(X, Y, polyDegree):
    m1,_ = X.shape
    m2,_ = Y.shape
    K = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            print(i,j)
            K[i,j] = (np.dot(X[i].T, Y[j]) + 1) ** polyDegree
    return K

if __name__ == '__main__':
    digitTrain = pd.read_csv("dataset/mnist_train.csv")
    digitTest = pd.read_csv("dataset/mnist_test.csv")

    # loading training set and test set
    data = applyPCA(digitTrain, digitTest)

    # computing kernel for training, once for every degree
    degree = 8
    for i in range(degree):
        kernelTrain = polyKernel(data["imgTrain"], data["imgTrain"], i+1)
        np.savetxt('k{0}.csv'.format(i+1), kernelTrain, delimiter='\n')
