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

def prediction(kernel, alpha, y):
    nSamples = kernel.shape[1]
    predictions = np.zeros(nSamples)

    for t in range(nSamples):
        predictions[t] = np.sum(alpha*kernel[:,t]) # this is the prediction
    #print(predictions)

    # if prediction is > 0 then return the label ?
    #return np.where(predictions > 0, self.classLabel, -1) if mapToClassLabels else predictions
    return predictions


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
        print(yTrain)

        for count in range(5): # given number of epochs
            loss = 0
            #trErr = 0

            # This is ONE EPOCH - a full cycle through data (each training point)
            for t in range(nSamples):
                # Predicting
                yHat = 1 if np.sum(alpha*yTrain*kernelTrain[:,t]) > 0 else -1
                #print(yTrain[t]) # 1 or -1

                # Updating weights
                if yHat != yTrain[t]:
                    alpha[t] += 1
                    loss += 1 # misclassified labels # da togliere, ottimizzare


            # Calculating epoch accuracy

            # Zero-one loss function on training data
            print("training accuracy")
            trainingaccuracy = "{:.2f}".format(1 - (float(loss)/nSamples))
            print(trainingaccuracy)

            # accuracy calculated on validation set
            print("validation accuracy")
            yPred = np.where(prediction(kernelVal, alpha, yVal.values) > 0, 1, -1)
            #yPred = np.where(prediction(kernelTrain, alpha) > 0, 1, -1)
            #print(yPred)

            #print(np.count_nonzero(yPred==yVal))
            #print(len(yVal))
            epochValaccuracy = np.count_nonzero(yPred==yVal) / float(len(yVal))
            #epochValaccuracy = np.count_nonzero(yPred==yTrain) / float(len(yTrain))
            print(epochValaccuracy)

            #print("my training error") loss function
            #print(trErr/nSamples)
            print()


        # Non zero weights and corresponding training image stored in object
        self.alphaCounters = alpha[np.nonzero(alpha)]
        self.supportVectors = xTrain.values[np.nonzero(alpha)]
        #print(self.alphaCounters)
        #print(self.supportVectors)


    def predict(self, xTest, yTest):
        print("... predicting ...")
        # Calculating kernel matrix
        kernelTest = polynomialKernel(self.supportVectors, xTest.values, self.hyperparameters)
        #print(kernelTest)

        # Calculating predicted y
        return prediction(kernelTest, self.alphaCounters, yTest.values)


    def classify(self, labels): # confronta le etichette con quelle che dovrebbero essere corrette, cioe quelle della classe (?)
        return np.where(labels == self.classLabel, 1, -1) # where assign 1 or -1 based on the condition (first parameter)
