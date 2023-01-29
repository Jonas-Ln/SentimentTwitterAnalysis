imprt numpy as np

#Input of x_train and test_data
class Neural_Network():

    def __init__(self,x_train,y_train):

    def forward_propagation(self, W1, b1, W2, b2):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) +b2
        A2 = softmax(A1)
        return Z1,A1,Z2,A2
    def back_propagation(self, Z1, A1, Z2, A2, W2, X, Y):
        m = Y.size
        dZ2 = A2
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, 2)
        dZ1 = W2.T.dot(dZ2) * get_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, 2)
        return dW1, db1, dW2, db2

    def get_ReLU(self,Z):
        return Z > 0

    def update_parameters(self, W1, b1, W2, b2, db1, dW2, db2, alpha):
        W1 = W1 - alpha *dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2

    def gradient_descent(self, X, Y, iterations, alpha):
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W2, X, Y)
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 50 == 0:
                print("Iteration: ",i)
                print("Accuracy ", get_accuracy(get_predictions(A2), Y))
        return

