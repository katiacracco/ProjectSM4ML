from multiClassKernelPerceptron import MultiClassKernelPerceptron

#import os
import numpy as np # mathematical functions
import pandas as pd # data analysis - CSV file I/O ??
import matplotlib.pyplot as plt # like MATLAB

# THIS IS THE KERNEL TO INITIALIZE FOR KERNEL PERCEPTRON
def polynomialKernel(x_i, x_j, power):
    return (x_i.T@x_j) ** power # cosa vuol dire ?



if __name__ == '__main__':
    digitTraining = pd.read_csv("../dataset/mnist_train.csv")
    #print(digit_training) # 60000 rows x 785 cols
    #print(type(digit_training)) returns <class 'pandas.core.frame.DataFrame'>

    # plot some training data
    for i in range(9):
        img = np.asarray(digitTraining.iloc[i,1:].values.reshape((28,28)))
        # asarray converts the input to an array
        # iloc is a purely integer-location based indexing for selection by positio
        plt.subplot(3,3,i+1) # nrows, ncols, index
        plt.imshow(img, cmap = 'gray')
    plt.show()
    # WANT TO END THE PROGRAM WITHOUT HAVING TO CLOSE MANUALLY THE IMAGE
    #plt.pause()
    #plt.close()


    for i in range(10):
        print("# Iteration {0}", i)
        MCKernelPerceptron = MultiClassKernelPerceptron(polynomialKernel, 3) # perchè 3 ?

        # LOAD DATA

        #data = MnistDigits(data_fname).get_split_datasets()
        #dataloaders = MnistDigitsPytorch.getDataLoader(data, batch_size=12)

        # Training model
        print("Training Kernel Perceptron")
        MCKernelPerceptron.train(data["images_train"], data["labels_train"], data["images_val"], data["labels_val"])
        # che parametri sono ?

        # Predicting with trained model
        print("Predicting Kernel Perceptron")
        yPred = MCKernelPerceptron.predict(data["images_test"])
        # che parametro è ?

        print("Results")
        print(yPred)
