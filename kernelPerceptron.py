import numpy as np

# kernel perceptron is one perceptron - multi class kernel perceptron append all the kernel perceptrons
class KernelPerceptron():
    def __init__(self, classLabel, kernel, hyperparameters=None):
        self.classLabel = classLabel
        self.kernel = kernel
        self.hyperparameters = hyperparameters

    # Training Algorithm Perceptron
    def train(self, xTrain, yTrain, xVal, yVal, kernelMatrixTrain=None, kernelMatrixVal=None):
        print("... training ...")

        # Setting training variables
        nSamples, nFeatures = xTrain.shape
        weights = np.zeros(nSamples) # shape return the dimension of xTrain (?)
        yTrain = self.classify(yTrain)
        yVal = self.classify(yVal)
        epochDone = 0 # epoch since val accuracy improvement
        bestVal = -1 # best val accuracy
        epoch = 0

        # Calculating kernel matrices (if not given in function call)
        #if kernelMatrixTrain is None:
        #    kernelMatrix = np.zeros((nSamples, nSamples)) # Setting weights to zero
        #    for i in range(nSamples): # COSA FA ?
        #        for j in range(nSamples):
        #            kernelMatrixTrain[i,j] = self.kernel(xTrain[i], xTrain[j], self.hyperparameters) # perchè hyperparameters ?
        #if kernelMatrixVal is None:
        #    kernelMatrix = np.zeros((nSamples, len(xVal))) # sono diversi questi due parametri?
        #    for i in range(nSamples):
        #        for j in range(len(xVal)):
        #            kernelMatrixVal[i,j] = self.kernel(xTrain[i], xVal[j], self.hyperparameters) # perchè hyperparameters ?

        # for each training point
        for tp in range(nSamples):
            # Predicting
            yHat = 1 if np.sum(weights*kernelMatrixTrain[:,t]) > 0 else 0 # or -1 ?
            # Updating weights
            if yHat != yTrain[t]:
                weights[t] += yTrain[t]

        # Non zero weights and corresponding training image stored in object
        self.savedWeights = weights[np.nonzero(weights)]
        self.supportVectors = xTrain[np.nonzero(weights)]

# COSA VOGLIONO DIRE?
#kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())

#for t in range(self.T):
#    for i in range(n_samples):
#        if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
#            self.alpha[i] += 1.0


    def predict(self, X, mapToClassLabels = True):
        # Calculating kernel matrix
        kernelMatrix = np.zeros((len(self.supportVectors), len(X))) # Setting weights to zero ??
        for i in range(len(self.supportVectors)): # COSA FA ?
            for j in range(len(X)):
                kernelMatrixTrain[i,j] = self.kernel(self.supportVectors[i], X[j], self.hyperparameters) # perchè hyperparameters ?

        # Calculating certainties [ALGORITHM]
        nSamples = kernelMatrix.shape[1]
        certainities = np.zeros(nSamples)
        for t in range(nSamples):
            certainities[t] = np.sum(self.savedWeights*kernelMatrix[:, t])

        return np.where(certainities > 0, self.classLabel, -1) if mapToClassLabels else certainties


    def classify(self, labels): # confronta le etichette con quelle che dovrebbero essere corrette, cioè quelle della classe (?)
        return np.where(labels == self.classLabel, 1, -1) # where assign 1 or -1 based on the condition (first parameter)
