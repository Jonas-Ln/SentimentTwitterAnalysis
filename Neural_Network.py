# Made by Jonas Lenz

import numpy as np

# Input of Training or Test Dataset as X and Labels as Y
# Always initiate with Neural_Network(Number of the Features)
# Then use the Gradient_Descent Function to Train a model
# Use the Make_a_prediction Function to Predict Y for a Test set

class Neural_Network:
    def __init__(self, feature_size):
        self.feature_size = feature_size
        # Random Seed to always get the same outcome for equal parameters -> testing Purposes
        np.random.seed(2)


    def forward_propagation(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def back_propagation(self, Z1, A1, A2, W2, X, Y):
        m = Y.size
        # Y to the fitting dimension (Here alrdy partly done but for another Task might be helpful)
        one_h_endoding = np.zeros((Y.max()+1, m))
        one_h_endoding[Y, np.arange(m)] = 1

        # Propagate from the Output to the Input Layer to get the Initial Values
        # After Subtracting what the actual Outcome should be with (Y)
        dZ2 = 2 * (A2 - one_h_endoding)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2,1)
        dZ1 = W2.T.dot(dZ2) * self.derivative_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1,1)
        return dW1, db1, dW2, db2

    # Derivative of the ReLu Function for Backpropagation
    def derivative_ReLU(self, Z):
        return Z > 0

    # Against Vanishing Gradient
    def ReLU(self, Z):
        return np.maximum(0, Z)

    # Output of Probabilities
    def softmax(self, Z):
        exp = np.exp(Z-np.max(Z))
        return exp / exp.sum(axis=0)

    def initialize_parameters(self):

        # Initialize the Parameters with the Feature size given in the initialization
        # Normalize each Feature
        W1 = np.random.normal(size = (10, self.feature_size)) * np.sqrt(1./self.feature_size)
        b1 = np.random.normal(size = (10, 1)) * np.sqrt(1./10)
        W2 = np.random.normal(size = (2, 10)) * np.sqrt(1./12)
        b2 = np.random.normal(size = (2, 1)) * np.sqrt(1./self.feature_size)
        return W1, b1, W2, b2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        # Update Parameters given by the Derivatives of the Back_propagation
        # And the Learning_Rate: Alpha
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * np.reshape(db1,(10,1))
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * np.reshape(db2,(2,1))
        return W1, b1, W2, b2

    def gradient_descent(self, X, Y, iterations, alpha):
        # Gradient Descent for getting step whise to the Optimum
        W1, b1, W2, b2 = self.initialize_parameters()
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.back_propagation(Z1, A1, Z2, A2,W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Accuracy ", self.get_accuracy(self.get_prediction(A2), Y))
                print(self.get_prediction(A2), Y)
        return W1, b1, W2, b2

    # Turn Probabilities into absolute Value 1 or 0
    def get_prediction(self, A2):
        return np.argmax(A2, 0)

    # Compare prediction to Result
    def get_accuracy(self, prediction, Y):
        return np.sum(prediction == Y) / Y.size

    # Get a prediction -> Ouptut: Absolute Value, Probability
    def make_a_prediction(self, X, W1, b1, W2, b2):
        _,_,_, A2 = self.forward_propagation(W1,b1,W2,b2,X)
        predictions = self.get_prediction(A2)
        return predictions, self.softmax(A2)