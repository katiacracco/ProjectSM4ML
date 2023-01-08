# ProjectSM4ML

Digit Classification with Kernel Perceptron

Author: Katia Cracco

preprocessing.py: this file pre-processes the dataset present in the dataset folder (zipped) by generating the files present in the input folder (zipped); 
  it also generates polynomial kernels which have not been loaded because they are too heavy.
  
digit.py: it contains the main of the principal program, that is of the algorithm to be implemented.

multiClassKernelPerceptron.py: this class contains the methods called by digit.py

kernelPerceptron.py: this class contains the methods called by multiClassKernelPerceptron.py
