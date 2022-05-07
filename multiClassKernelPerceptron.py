from kernelPerceptron import KernelPerceptron, polynomialKernel

import numpy as np


# One vs all multi class kernel perceptron
class MultiClassKernelPerceptron():
    def __init__(self, epochNumber, polynomialDegree): # se inserisco kernel in self evito di calcolarlo ogni volta?
        self.epochNumber = epochNumber
        self.polynomialDegree = polynomialDegree
        self.perceptrons = []
        self.predictorsAverage = []
        self.smallestErrorPredictor = []
        self.accuracy = []

    def train(self, xTrain, yTrain):
        # Creating one perceptron for each unique label (10 digits)
        for label in np.unique(yTrain):
            self.perceptrons.append(KernelPerceptron(label, self.epochNumber, self.polynomialDegree))

        # Kernel matrix are same for all classes so calc once and pass around
        kernelTrain = polynomialKernel(xTrain.values, xTrain.values, self.polynomialDegree)
        #print(kernelTrain)

        # Training models (10 binary classifiers) - one vs all encoding
        for x in self.perceptrons:
            x.train(xTrain, yTrain, kernelTrain) # xTrain è sempre uguale o shuffle?

    def predict(self, xTest, yTest):
        # matrix 4-dim: (((xTest * interim) * 2 versions) * 10 perceptrons)
        interim = int(self.epochNumber/5)
        perceptronPredictions = np.zeros((2,interim,len(xTest),len(self.perceptrons)))

        # Each model gives certainty that image belongs to its class
        for i, perceptron in enumerate(self.perceptrons): # i = 0,1,2,3..9
            perceptronPredictions[:,:,:,i] = perceptron.predict(xTest, yTest)

        #print(perceptronPredictions.shape)
        #print(perceptronPredictions)


        maxPrediction = np.zeros((2,len(xTest),interim))

        # Index/label of perceptron with max certainty
        for version in range(2):
            for inter in range(interim):
                maxPrediction[version,:,inter] = np.argmax(perceptronPredictions[version,inter,:,:], axis = 1) # argmax along cols, return argmax for every row
        #print(maxPrediction.shape)
        #print(maxPrediction)
        #print(maxPrediction.shape[0])
        #res = np.array()
        """
        for i in range(maxPrediction.shape[0]):
            for j in range(maxPrediction.shape[2]):
                for a in maxPrediction[i,:,j]:
                    print(a)
                    print()
        """


        # Return class label for most accurate prediction of perceptron for each image
        return np.array([[self.perceptrons[int(a)].classLabel for a in maxPrediction[i,:,j]]
                                for j in range(maxPrediction.shape[2]) for i in range(maxPrediction.shape[0])])

    def statistics(self):
        # CHIEDERE CONFERMA SE POSSO CONSIDERARE ALPHA AL POSTO DI W COME PREDICTORS
        #loss = np.sum(alpha)

        #average = loss / (nSamples*self.epochNumber)
        print("The average of the predictors in the ensemble is {:.2f}".format(average))
        #smallest = np.min(self.alphaCounters) # non ha senso come valore secondo me
        print("The predictor achieving the smallest training error among those in the ensemble is {0}".format(smallest))

        # Saving accuracy for each classifier
        #self.predictorsAverage.append(average)
        #self.smallestErrorPredictor.append(smallest)
        #self.accuracy.append(1 - (float( loss / (nSamples*self.epochNumber) )))

        return
