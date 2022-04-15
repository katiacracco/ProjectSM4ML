from multiClassKernelPerceptron import MultiClassKernelPerceptron

#import os
import numpy as np # mathematical functions
import pandas as pd # data analysis - CSV file I/O ??
import matplotlib.pyplot as plt # like MATLAB
import torch

#from torch.utils.data import Dataset



# THIS IS THE KERNEL TO INITIALIZE FOR KERNEL PERCEPTRON
def polynomialKernel(x, y, power):
    return (np.dot(x, y.T) + 1) ** power


def getDataset(data):
    nSamples = data.shape[0]
    divider = int(nSamples * 0.2)
    np.random.shuffle(data) # SERVE?

    labels = data.label
    digits = data.drop(['label'], axis=1)

    return {"imgVal": digits[:divider],
            "imgTrain": digits[divider:],
            "labelVal": labels[:divider],
            "labelTrain": labels[divider:]
    }

def getDataLoader(dataset):



if __name__ == '__main__':
    #digitTraining = torchvision.datasets.MNIST(root:"../dataset", train: True, download: False)
    trainData = pd.read_csv("../dataset/mnist_train.csv")
    print(trainData) # 60000 rows x 785 cols

#    td = np.genfromtxt("../dataset/small.dat")
#    print(td)
    #print(type(digit_training)) returns <class 'pandas.core.frame.DataFrame'>


    # plot some training data
    for i in range(9):
        img = np.asarray(trainData.iloc[i,1:].values.reshape((28,28)))
        # asarray converts the input to an array
        # iloc is a purely integer-location based indexing for selection by positio
        plt.subplot(3,3,i+1) # nrows, ncols, index
        plt.imshow(img, cmap = 'gray')
    plt.show()
    # WANT TO END THE PROGRAM WITHOUT HAVING TO CLOSE MANUALLY THE IMAGE
    #plt.pause()
    #plt.close()

"""
    for i in range(10):
        print("# Iteration {0}", i)
        MCKernelPerceptron = MultiClassKernelPerceptron(polynomialKernel, 3) # perch 3 ?

        # LOAD DATA
        #data = getDataset(trainData)
        #dataloaders = getDataLoader(data)
        #trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True) # is necessario avere 3 dataloaders? train - val - test

        #data = MnistDigits(data_fname).get_split_datasets()
        #dataloaders = MnistDigitsPytorch.getDataLoader(data, batch_size=12)

        # Training model
        print("Training Kernel Perceptron")
        MCKernelPerceptron.train(data["images_train"], data["labels_train"], data["images_val"], data["labels_val"])
        # che parametri sono ?

        # Predicting with trained model
        print("Predicting Kernel Perceptron")
        yPred = MCKernelPerceptron.predict(data["images_test"])
        # che parametro is ?

        print("Results")
        print(yPred)
"""
