

import numpy as np

# Input of x_train and test_data
class Neural_Network:
    def __init__(self, feature_size):
        self.feature_size = feature_size
        np.random.seed(2)


    def forward_propagation(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def back_propagation(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        m = Y.size
        #encode_y
        one_h_endoding = np.zeros((Y.max()+1, m))
        one_h_endoding[Y, np.arange(m)] = 1
        #encode_y = np.zeros((Y.size, Y.max()+1))
        #enocde_y[np.arange(Y.size),Y] = 1
        dZ2 = 2 * (A2 - one_h_endoding)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2,1)
        dZ1 = W2.T.dot(dZ2) * self.derivative_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1,1)
        return dW1, db1, dW2, db2

    def derivative_ReLU(self, Z):
        return Z > 0

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp = np.exp(Z-np.max(Z))
        return exp / exp.sum(axis=0)

    def initialize_parameters(self):
        W1 = np.random.normal(size = (10, self.feature_size)) * np.sqrt(1./self.feature_size)
        b1 = np.random.normal(size = (10, 1)) * np.sqrt(1./10)
        W2 = np.random.normal(size = (2, 10)) * np.sqrt(1./12)
        b2 = np.random.normal(size = (2, 1)) * np.sqrt(1./self.feature_size)
        #print(W1, b1, W2, b2)
        return W1, b1, W2, b2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * np.reshape(db1,(10,1))
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * np.reshape(db2,(2,1))

        return W1, b1, W2, b2

    def gradient_descent(self, X, Y, iterations, alpha):
        size, m = X.shape
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

    def get_prediction(self, A2):
        return np.argmax(A2, 0)


    def get_accuracy(self, prediction, Y):
        return np.sum(prediction == Y) / Y.size

    def make_a_prediction(self, X, W1, b1, W2, b2):
        _,_,_, A2 = self.forward_propagation(W1,b1,W2,b2,X)
        predictions = self.get_prediction(A2)
        return predictions, self.softmax(A2)