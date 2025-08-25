import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  fx = sigmoid(x)
  return fx * (1 - fx)