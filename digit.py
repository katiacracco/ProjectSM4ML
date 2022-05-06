from multiClassKernelPerceptron import MultiClassKernelPerceptron

import numpy as np # mathematical functions
import pandas as pd # data analysis - CSV file I/O ??
import matplotlib.pyplot as plt # like MATLAB
import sys

def getDataset(data1, data2):
    # training data
    size = 1000
    sizeTest = int(size/2)
    divider = int(size * 0.2) # 4000

    data1 = data1.sample(frac=1).reset_index(drop=True) # problema che mescoli train e val?
    #sh = data1.reindex(np.random.permutation(data1.index))

    # test data
    #data2 = data2.sample(frac=1).reset_index(drop=True) # non credo serva mescolare il test

    labels1 = data1.label # <class 'pandas.core.series.Series'>
    digits1 = data1.drop(['label'], axis=1) # <class 'pandas.core.frame.DataFrame'>

    labels2 = data2.label
    digits2 = data2.drop(['label'], axis=1)

    return {"imgVal": digits1[:divider],
            "imgTrain": digits1[divider:size],
            "imgTest": digits2[:sizeTest],
            "labelVal": labels1[:divider],
            "labelTrain": labels1[divider:size],
            "labelTest": labels2[:sizeTest]
    }


class DigitsDataset(torch.utils.data.Dataset):
    def __init__(self, data, mode):
        if mode == "train":
            self.labels = data["labelTrain"]
            self.imgs = data["imgTrain"]
        if mode == "val":
            self.labels = data["labelVal"]
            self.imgs = data["imgVal"]
        if mode == "test":
            self.labels = data["labelTest"]
            self.imgs = data["imgTest"]

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


"""
def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # CNN takes 2D not 1D input (unlike other models in this repo)
        image = self.images[idx].reshape(1, MnistDigits.image_size, MnistDigits.image_size)
        return image.astype(np.float32), self.labels[idx]
"""



def getDataLoader(dataset): # or 12 ?
    data = {mode: DigitsDataset(dataset,mode) for mode in ["train", "val", "test"]}

    return {
        "train": torch.utils.data.DataLoader(
            data["train"],
            batch_size=12,
            shuffle=True
        ),
        "val": torch.utils.data.DataLoader(
            data["val"],
            batch_size=12,
            shuffle=False
        ),
        "test": torch.utils.data.DataLoader(
            data["test"],
            batch_size=12,
            shuffle=False
        )
    }


if __name__ == '__main__':
    digitTrain = pd.read_csv("../dataset/mnist_train.csv") # type pandas.core.frame.DataFrame
    digitTest = pd.read_csv("../dataset/mnist_test.csv")

    #print("# Iteration {0}".format(i))
    MCKernelPerceptron = MultiClassKernelPerceptron(3) # perch 3 ?

    # LOAD DATA
    data = getDataset(digitTrain, digitTest)
    dataloaders = getDataLoader(data)
    #print(data["imgTrain"])

    # Training model
    print("Training Kernel Perceptron")
    MCKernelPerceptron.train(data["imgTrain"], data["labelTrain"], data["imgVal"], data["labelVal"])

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
