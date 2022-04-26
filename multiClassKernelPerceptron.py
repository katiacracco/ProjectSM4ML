from kernelPerceptron import KernelPerceptron, polynomialKernel

import numpy as np



# One vs all multi class kernel perceptron
class MultiClassKernelPerceptron():
    def __init__(self, hyperparameters): # se inserisco kernel in self evito di calcolarlo ogni volta?
        self.hyperparameters = hyperparameters
        self.perceptrons = []

    def train(self, xTrain, yTrain, xVal, yVal):
        # Creating one perceptron for each unique label (10 digits)
        for label in np.unique(yTrain):
            self.perceptrons.append(KernelPerceptron(label, self.hyperparameters))

        # Kernel matrix are same for all classes so calc once and pass around

        kernelTrain = polynomialKernel(xTrain.values, xTrain.values, self.hyperparameters)
        print(kernelTrain)

        kernelVal = polynomialKernel(xTrain.values, xVal.values, self.hyperparameters)
        print(kernelVal)

        # Training models (10 binary classifiers) - one vs all encoding
        for x in self.perceptrons:
            x.train(xTrain, yTrain, xVal, yVal, kernelTrain, kernelVal)

    def predict(self, X):
        # Each model gives certainty that image belongs to its class
        perceptronCertainities = np.zeros((len(X), len(self.perceptrons))) # X rows - 10 cols
        for i, perceptron in enumerate(self.perceptrons):
            perceptronCertainities[:,i] = perceptron.predict(X, mapToClassLabels=False)

        # Index of perceptron with max certainty
        indexMaxCertainity = np.argmax(perceptronCertainities, axis = 1)

        # Return class label for most certain perceptron for each image
        return np.array([self.perceptrons[i].classLabel for i in indexMaxCertainity])
