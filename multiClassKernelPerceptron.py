from kernelPerceptron import KernelPerceptron, polynomialKernel

import numpy as np


# One vs all multi class kernel perceptron
class MultiClassKernelPerceptron():
    def __init__(self, epochNumber, polynomialDegree): # se inserisco kernel in self evito di calcolarlo ogni volta?
        self.epochNumber = epochNumber
        self.polynomialDegree = polynomialDegree
        self.perceptrons = []

    def train(self, xTrain, yTrain):
        # Creating one perceptron for each unique label (10 digits)
        for label in np.unique(yTrain):
            self.perceptrons.append(KernelPerceptron(label, self.polynomialDegree))

        # Kernel matrix are same for all classes so calc once and pass around
        kernelTrain = polynomialKernel(xTrain.values, xTrain.values, self.polynomialDegree)
        #print(kernelTrain)

        # Training models (10 binary classifiers) - one vs all encoding
        for x in self.perceptrons:
            x.train(xTrain, yTrain, kernelTrain) # xTrain è sempre uguale o shuffle?

    def predict(self, xTest, yTest):
        # Each model gives certainty that image belongs to its class
        perceptronPredictions = np.zeros((len(xTest), len(self.perceptrons))) # X rows - 10 cols
        for i, perceptron in enumerate(self.perceptrons): # i = 0,1,2,3..9
            perceptronPredictions[:,i] = perceptron.predict(xTest, yTest)
            #print(perceptronPredictions[:,i])

        # Index of perceptron with max certainty
        maxPrediction = np.argmax(perceptronPredictions, axis = 1) # argmax along cols, return argmax for every row
        #print(maxPrediction)

        # Return class label for most accurate prediction of perceptron for each image
        return np.array([self.perceptrons[i].classLabel for i in maxPrediction])
