import numpy as np
from utils import sigmoid


class ThreeBlue1BrownNeuralNet: 
    '''
    A neural network for MNIST with:
        - 784 inputs (28x28 image)
        - a first hidden layer with 64 neurons
        - a second hidden layer with 32 neurons
        - an output layer with 10 neurons (for digits 0-9)
    '''
    def __init__(self):
        input_size = 784
        hidden1_size = 16
        hidden2_size = 16
        output_size = 10

        self.w1 = np.random.randn(hidden1_size, input_size)
        self.b1 = np.zeros(hidden1_size)

        self.w2 = np.random.randn(hidden2_size, hidden1_size)
        self.b2 = np.zeros(hidden2_size)

        self.w_out = np.random.randn(output_size, hidden2_size)
        self.b_out = np.zeros(output_size)

    def feedforward(self, x):
        x = x.flatten()

        h1 = sigmoid(np.dot(self.w1, x) + self.b1)

        h2 = sigmoid(np.dot(self.w2, h1) + self.b2)

        output = sigmoid(np.dot(self.w_out, h2) + self.b_out)

        return output