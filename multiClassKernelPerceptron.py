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

        # GET A BASTCH OF TRAINING DATA WITH ENUMERATE
        # Training models (10 binary classifiers) - one vs all encoding
        for x in self.perceptrons:
            x.train(xTrain, yTrain, xVal, yVal, kernelTrain, kernelVal) # xTrain è sempre uguale o shuffle?

    def predict(self, X):
        # Each model gives certainty that image belongs to its class
        perceptronPredictions = np.zeros((len(X), len(self.perceptrons))) # X rows - 10 cols
        for i, perceptron in enumerate(self.perceptrons): # i = 0,1,2,3..9
            perceptronPredictions[:,i] = perceptron.predict(X, mapToClassLabels=False)
            #print(perceptronPredictions[:,i])

        # Index of perceptron with max certainty
        maxPrediction = np.argmax(perceptronPredictions, axis = 1) # argmax along cols, return argmax for every row
        #print(maxPrediction)

        # Return class label for most accurate prediction of perceptron for each image
        return np.array([self.perceptrons[i].classLabel for i in maxPrediction])
