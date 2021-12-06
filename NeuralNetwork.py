import numpy as np
import pandas as pd

mydata= pd.read_csv("millingdata.csv")
# Input Parameter = (Milling speed (rpm), Particle size(microns) ,Milling time (h))
X = mydata[['Milling speed (rpm)','Particle size ','Milling time (h)']]
# Output Parameter = (coating thickness in microns)
y = mydata[['Measured']]

# Normalization of the data set
X = X/np.amax(X, axis=0) #maximum of X array
y = y/np.amax(y, axis=0) # maximum of y array

class NeuralNetwork(object):
    def __init__(self):
        # Enter Number of input layer  neurons
        self.inputSize = 3
        # Enter Number of  hidden layer neurons
        self.hiddenSize = 5
        # Enter Number of output layer neurons
        self.outputSize = 1
        # Enter the value of learning rate
        self.eta = 0.5
        
        #weights 
        self.v = np.random.randn(self.inputSize, self.hiddenSize)   #  weight matrix from input to hidden layer
        self.W = np.random.randn(self.hiddenSize, self.outputSize)  # weight matrix from hidden to output layer
        
    def feedForward(self, X):
        #forward propogation through the network
        self.I_H = np.dot(X, self.v) # Finding the Input of Hidden layer neurons with dot product of X (input) and v- weights
        self.O_H = self.sigmoid(self.I_H) # Output of Hidden layer neurons with transfer function as log sigmoid
        self.I_O = np.dot(self.O_H, self.W) # Input of Output Layer neurons with dot product of hidden layer (O_H) and w- weights 
        O_O = self.sigmoid(self.I_O)  # Output of output layer neurons with transfer function as log sigmoid
        return O_O
        
    def sigmoid(self, s, deriv=False):
        
        if (deriv == True):
            return s * (1 - s)   # Define the derivative of log sigmoid function
        return 1/(1 + np.exp(-s))     # define the log sigmoid function
    
    def backward(self, X, y, O_O):
        #Weight update by backward propogate through the network
        self.O_O_error = y - O_O # error in output of output layer from target value
        self.O_O_delta = self.O_O_error * self.sigmoid(O_O, deriv=True)
        
        self.O_H_error = self.O_O_delta.dot(self.W.T) #
        self.O_H_delta = self.O_H_error * self.sigmoid(self.O_H, deriv=True) 

        self.W += self.eta* self.O_H.T.dot(self.O_O_delta)# Apply formula for w-weight updatation
        self.v += self.eta* X.T.dot(self.O_H_delta) # Apply formula for v-weight updatation
        
        
    def training(self, X, y):
        O_O = self.feedForward(X)
        self.backward(X, y, O_O)
        
NN = NeuralNetwork()

                                      
        
print("Input:\n " + str(X))
print("Target:\n " + str(267*y)) # since 267 is the largest value in output data

for i in range(1000): # Number of iteration
    
    if (i % 100 == 0):    # Number of testing data
        
        print("MeanSquareError after every 100 step:" + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.training(X, y)
    NN.feedForward(X)
print("Mean Square Error:" + str(np.mean(np.square(y - NN.feedForward(X)))))    
print("Predicted Output: \n" + str(267*(NN.feedForward(X))))
print('Error in Prediction :' + str(y - NN.feedForward(X)))

