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
        self.classLabel = classLabel # label for the specific classifier
        self.hyperparameters = hyperparameters

    # Training Algorithm Perceptron
    def train(self, xTrain, yTrain, xVal, yVal, kernelTrain, kernelVal):
        print("... training classifier {0} ...".format(self.classLabel))

        # Setting training variables
        nSamples, nFeatures = xTrain.shape
        alpha = np.zeros(nSamples) # shape return the dimension of xTrain (?)
        yTrain = self.classify(yTrain) # highlight the same labels as the current perceptron
        #yValBin = self.classify(yVal) # 1 if label corresponds
        #epochDone = 0 # epoch since val accuracy improvement
        #bestVal = -1 # best val accuracy
        #epoch = 0
        sum = 0
        flag = True

        #while flag:
        for count in range(5): # given number of epochs
            #S = []
            #sum = 0
            #update = 0
            #print(S)
            trErr = 0

            # This is ONE EPOCH - a full cycle through data (each training point)
            for t in range(nSamples):
                #print(alpha)
                # Predicting
                #for s in S:
                    #print(yTrain[s])
                    #print(kernelTrain[s,t])
                    #print(yTrain[s]*kernelTrain[s,t])
                #    sum += yTrain[s]*kernelTrain[s,t] # yTrain deve essere 0-1 o il valore dell'etichetta?
                #yHat = 1 if sum > 0 else -1
                yHat = 1 if np.sum(alpha*yTrain*kernelTrain[:,t]) > 0 else -1
                #print(yTrain[t]) # 1 or -1

                # Updating weights
                if yHat != yTrain[t]:
                    alpha[t] += 1
                    out = yHat - yTrain[t]
                    trErr += out
                    #weights[t] += yTrain[t] # WEIGHTS COSA DOVREBBERO RAPPRESENTARE ?
                    #S.append(t)
                    #update += 1
            print(trErr/nSamples)


            # if in the last epoch there were no unpdate, break
            #if update == 0:
            #    flag = False

        # Non zero weights and corresponding training image stored in object
        self.alphaCounters = alpha[np.nonzero(alpha)]
        self.supportVectors = xTrain.values[np.nonzero(alpha)]
        #self.alphaCounters = S
        #self.supportVectors = xTrain.values[np.nonzero(S)]

        print(self.alphaCounters)
        #print(self.supportVectors)

        # calcolare accurancy???


    def predict(self, X, mapToClassLabels = True):
        print("... predicting ...")
        # Calculating kernel matrix
        kernel = polynomialKernel(self.supportVectors, X.values, self.hyperparameters)
        #print(kernel)

        # Calculating certainties [ALGORITHM]
        nSamples = kernel.shape[1]
        predictions = np.zeros(nSamples)
        for t in range(nSamples):
            predictions[t] = np.sum(self.alphaCounters*kernel[:, t]) # this is the prediction
        #print(predictions)

        # if prediction is >0 then return the label ?
        #return np.where(predictions > 0, self.classLabel, -1) if mapToClassLabels else predictions
        return predictions


    def classify(self, labels): # confronta le etichette con quelle che dovrebbero essere corrette, cioe quelle della classe (?)
        return np.where(labels == self.classLabel, 1, -1) # where assign 1 or -1 based on the condition (first parameter)
