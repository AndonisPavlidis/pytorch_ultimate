import numpy as np

class NeuralNetworkFromScratch:
    """
    A class representing a neural network implemented from scratch.

    Args:
        learning_rate: The learning rate for the optimizer.
        X_train: The input training data.
        y_train: The target training data.
        X_test: The input test data.
        y_test: The target test data.

    Attributes:
        weights: The weights of the neural network.
        bias: The bias of the neural network.
        learning_rate: The learning rate for the optimizer.
        X_train: The input training data.
        y_train: The target training data.
        X_test: The input test data.
        y_test: The target test data.
        training_losses: The training loss at each iteration.
        test_losses: The test loss at each iteration.
    """

    def __init__(self, learning_rate: float, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
        self.weights = np.random.randn(X_train.shape[1])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []

    def activation(self, X: np.array) -> np.array:
        """
        Applies the sigmoid activation function to the input.

        Args:
            x: The input to the activation function.

        Returns:
            The output of the activation function.
        """
        return 1 / (1 + np.exp(-X))

    def dactivation(self, X: np.array) -> np.array:
        """
        Computes the derivative of the sigmoid activation function.

        Args:
            x: The input to the activation function.

        Returns:
            The derivative of the activation function.
        """
        return self.activation(X) * (1 - self.activation(X))

    def forward(self, X):
        """
        Performs the forward pass of the neural network.

        Args:
            X: The input data.

        Returns:
            The output of the neural network.
        """
        hidden_1 = np.dot(X, self.weights) + self.bias
        activate_1 = self.activation(hidden_1)
        return activate_1

    def backward(self, X, y_true):
        """
        Performs the backward pass of the neural network and calculates the gradients.

        Args:
            X: The input data.
            y_true: The true labels.

        Returns:
            Gradients with respect to the bias and weights.
        """
        y_pred = self.forward(X)
        dL_dpred = 2 * (y_pred - y_true)
        
        hidden_1 = np.dot(X, self.weights) + self.bias
        dpred_dhidden1 = self.dactivation(hidden_1)
        
        dhidden1_db = 1
        dhidden1_dw = X
        

        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
        return dL_db, dL_dw

    def optimizer(self, dL_db, dL_dw):
        """
        Updates the weights and bias of the neural network using the gradients.

        Args:
            dL_db (numpy.ndarray): The gradient with respect to the bias.
            dL_dw (numpy.ndarray): The gradient with respect to the weights.
        """
        self.bias = self.bias - dL_db * self.learning_rate
        self.weights = self.weights - dL_dw * self.learning_rate

    def train(self, ITERATIONS):
        """
        Trains the neural network for a specified number of iterations.

        Args:
            ITERATIONS (int): The number of training iterations.

        Returns:
            str: A message indicating that the training has finished successfully.
        """
        for i in range(ITERATIONS):
            random_pos = np.random.randint(len(self.X_train))
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])
            L = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(L)
            dL_db, dL_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])
            self.optimizer(dL_db, dL_dw)
            L_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.square(y_pred - y_true)
            self.L_test.append(L_sum)
        return "Training successfully finished"
