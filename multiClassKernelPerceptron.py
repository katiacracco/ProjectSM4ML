from kernelPerceptron import KernelPerceptron, polynomialKernel

import pandas as pd
import numpy as np

# One vs all multi class kernel perceptron
class MultiClassKernelPerceptron():
    def __init__(self, epochNumber, polynomialDegree):
        self.epochNumber = epochNumber
        self.polynomialDegree = polynomialDegree
        self.perceptrons = []

    def train(self, xTrain, yTrain):
        # Creating one perceptron for each unique label (10 digits)
        for label in np.unique(yTrain):
            self.perceptrons.append(KernelPerceptron(label, self.epochNumber, self.polynomialDegree))

        # Kernel matrix is the same for all classes so it is calculated once and pass around
        kernelTrain = pd.read_csv("../input/k{0}.csv".format(self.polynomialDegree), header=None)
        kernelTrain = np.reshape(kernelTrain.values, (10000,10000))

        # Training models (10 binary classifiers)
        for x in self.perceptrons:
            x.train(xTrain.values, yTrain.values, kernelTrain)

    def predict(self, xTest, yTest):
        interim = int(self.epochNumber/5)
        perceptronPredictions = np.zeros((2,interim,len(xTest),len(self.perceptrons)))

        # Each model gives the certainty that an image belongs to its class
        for i, perceptron in enumerate(self.perceptrons):
            perceptronPredictions[:,:,:,i] = perceptron.predict(xTest, yTest)

        maxPrediction = np.zeros((2,len(xTest),interim))

        # Index/label of perceptron with max certainty
        for version in range(2):
            for inter in range(interim):
                maxPrediction[version,:,inter] = np.argmax(perceptronPredictions[version,inter,:,:], axis = 1)

        # Return class label for most accurate prediction of perceptron for each image
        return np.array([[self.perceptrons[int(a)].classLabel for a in maxPrediction[i,:,j]]
                                for j in range(maxPrediction.shape[2]) for i in range(maxPrediction.shape[0])])
