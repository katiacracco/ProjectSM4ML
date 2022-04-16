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




class DigitsDataset(Dataset):
    def __init__(self, data, mode):
        if mode == "train":
            self.labels = data["labelTrain"]
            self.imgs = data["imgTrain"]
        if mode == "val":
            self.labels = data["labelVal"]
            self.imgs = data["imgVal"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # DA RIVEDERE
        file = pd.read_csv("../dataset/mnist_train.csv")
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        imgPath = file.iloc[idx,1:]
        image = read_image(imgPath)
        #label = self.img_labels.iloc[idx, 0]
        label = file.iloc[idx,0]
        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return image, label


def getDataLoader(dataset, batch=64): # or 12 ?
    data = {mode: DigitsDataset(data,mode) for mode in ["train", "val"]}

    return {
        "train": torch.utils.data.DataLoader(
            data["train"],
            batchSize=batch,
            shuffle=True,
            dropLast= True,
        ),
        "val": torch.utils.data.DataLoader(
            data["val"],
            batch_size=batch,
            shuffle=False,
            dropLast= True
        )
    }



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


    for i in range(10):
        print("# Iteration {0}", i)
        MCKernelPerceptron = MultiClassKernelPerceptron(polynomialKernel, 3) # perch 3 ?

        # LOAD DATA
        data = getDataset(trainData)
        dataloaders = getDataLoader(data)

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
