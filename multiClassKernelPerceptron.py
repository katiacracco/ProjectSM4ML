from kernelPerceptron import KernelPerceptron

import numpy as np

# polynomial kernel to use predicting y
def poly(X, Y, power):
    m1,_ = X.shape
    m2,_ = Y.shape
    K = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            #print("{0} {1}".format(i, j))
            K[i,j] = (1 + np.dot(X[i].T, Y[j])) ** power # +1 va tolto?
            #print(K[i,j])
    return K


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

        kernelTrain = poly(xTrain.values, xTrain.values, self.hyperparameters)
        print(kernelTrain)

        kernelVal = poly(xTrain.values, xVal.values, self.hyperparameters)
        print(kernelVal)

        # Training models (10 binary classifiers) - one vs all encoding
        for x in self.perceptrons:
            x.train(xTrain, yTrain, xVal, yVal, kernelTrain, kernelVal)

    def predict(self, X):
        # Each model gives certainty that image belongs to its class
        perceptronCertainities = np.zeros(len(X), len(self.perceptrons))
        for i, perceptron in enumerate(self.perceptrons):
            perceptronCertainities[:,i] = perceptron.predict(X, mapToClassLabels=False)

        # Index of perceptron with max certainty
        indexMaxCertainity = np.argmax(perceptronCertainities, axis = 1)

        # Return class label for most certain perceptron for each image
        return np.array([self.perceptrons[i].classLabel for i in indexMaxCertainity])
