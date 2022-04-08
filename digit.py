#import os
import numpy as np # mathematical functions
import pandas as pd # data analysis - CSV file I/O ??
import matplotlib.pyplot as plt # like MATLAB

digit_training = pd.read_csv("dataset/mnist_train.csv")
#print(digit_training) # 60000 rows x 785 cols
#print(type(digit_training)) returns <class 'pandas.core.frame.DataFrame'>

# plot some training data
for i in range(9):
    img = np.asarray(digit_training.iloc[i,1:].values.reshape((28,28)))
    # asarray converts the input to an array
    # iloc is a purely integer-location based indexing for selection by positio
    plt.subplot(3,3,i+1) # nrows, ncols, index
    plt.imshow(img, cmap = 'gray')
plt.show()
# WANT TO END THE PROGRAM WITHOUT HAVING TO CLOSE MANUALLY THE IMAGE





#plt.pause()
#plt.close()
