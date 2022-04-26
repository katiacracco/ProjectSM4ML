import numpy as np

# polynomial kernel to use predicting y
def polynomialKernel(X, Y, power):
    m1,_ = X.shape
    m2,_ = Y.shape
    K = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            #print("{0} {1}".format(i, j))
            K[i,j] = (1 + np.dot(X[i].T, Y[j])) ** power # +1 va tolto?
            #print(K[i,j])
    return K


# kernel perceptron is one perceptron - multi class kernel perceptron append all the kernel perceptrons
class KernelPerceptron():
    def __init__(self, classLabel, hyperparameters):
        self.classLabel = classLabel
        self.hyperparameters = hyperparameters

    # Training Algorithm Perceptron
    def train(self, xTrain, yTrain, xVal, yVal, kernelTrain, kernelVal):
        print("... training ...")

        # Setting training variables
        nSamples, nFeatures = xTrain.shape
        weights = np.zeros(nSamples) # shape return the dimension of xTrain (?)
        yTrain = self.classify(yTrain) # highlight the same labels as the current perceptron
        yVal = self.classify(yVal) # 1 if label corresponds
        epochDone = 0 # epoch since val accuracy improvement
        bestVal = -1 # best val accuracy
        epoch = 0
        S = []
        y = 0
        flag = True

        while flag:
            update = 0

            # This is ONE EPOCH - a full cycle through data (each training point)
            for t in range(nSamples):
                # Predicting
                #for s in S:
                    #print(yTrain[s])
                    #print(kernelTrain[s,t])
                    #print(yTrain[s]*kernelTrain[s,t])
                    #y += yTrain[s]*kernelTrain[s,t]
                #yHat = 1 if y > 0 else -1
                yHat = 1 if np.sum(weights*kernelTrain[:,t]) > 0 else -1

                # Updating weights
                if yHat != yTrain[t]:
                    weights[t] += yTrain[t]
                    #S.append(t)
                    update += 1

            # if in the last epoch there wew no unpdate, break
            if update > 0:
                flag = False

        # Non zero weights and corresponding training image stored in object
        self.savedWeights = weights[np.nonzero(weights)]
        self.supportVectors = xTrain.values[np.nonzero(weights)]

        # calcolare accurancy???


    def predict(self, X, mapToClassLabels = True):
        print("... predicting ...")
        # Calculating kernel matrix
        kernel = polynomialKernel(self.supportVectors, X.values, self.hyperparameters)
        print(kernel)

        # Calculating certainties [ALGORITHM]
        nSamples = kernel.shape[1]
        certainities = np.zeros(nSamples)
        for t in range(nSamples):
            certainities[t] = np.sum(self.savedWeights*kernel[:, t])

        return np.where(certainities > 0, self.classLabel, -1) if mapToClassLabels else certainities


    def classify(self, labels): # confronta le etichette con quelle che dovrebbero essere corrette, cioe quelle della classe (?)
        return np.where(labels == self.classLabel, 1, -1) # where assign 1 or -1 based on the condition (first parameter)
