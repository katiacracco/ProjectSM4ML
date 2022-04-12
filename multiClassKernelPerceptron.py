from kernelPerceptron import KernelPerceptron

import numpy as np

# One vs all multi class kernel perceptron
class MultiClassKernelPerceptron():
    def __init__(self, kernel, hyperparameters):
        self.kernel = kernel
        self.hyperparameters = hyperparameters
        self.perceptrons = []

    def train(self, xTrain, yTrain, xVal, yVal):
        # Creating one perceptron for each unique label (10 digits)
        for label in np.unique(yTrain):
            self.perceptrons.append(KernelPerceptron(label, self.kernel, self.hyperparameters))

        # Kernel matrix are same for all classes so calc once and pass around
        kernelMatrix = np.zeros((nSamples, nSamples)) # Setting weights to zero
        for i in range(nSamples): # COSA FA ?
            for j in range(nSamples):
                kernelMatrixTrain[i,j] = self.kernel(xTrain[i], xTrain[j], self.hyperparameters) # perchè hyperparameters ?
        kernelMatrix = np.zeros((nSamples, len(xVal))) # sono diversi questi due parametri?
        for i in range(nSamples):
            for j in range(len(xVal)):
                kernelMatrixVal[i,j] = self.kernel(xTrain[i], xVal[j], self.hyperparameters) # perchè hyperparameters ?

        # Training models (each of 10 perceptrons)
        for x in self.perceptrons:
            x.train(xTrain, yTrain, xVal, yVal, kernelMatrixTrain, kernelMatrixVal)

    def predict(self, X):
        # Each model gives certainty that image belongs to its class
        perceptronCertainities = np.zeros(len(X), len(self.perceptrons))
        for i, perceptron in enumerate(self.perceptrons):
            perceptronCertainities[:,i] = perceptron.predict(X, mapToClassLabels=False)

        # Index of perceptron with max certainty
        indexMaxCertainity = np.argmax(perceptronCertainities, axis = 1)

        # Return class label for most certain perceptron for each image
        return np.array([self.perceptrons[i].classLabel for i in indexMaxCertainity])
