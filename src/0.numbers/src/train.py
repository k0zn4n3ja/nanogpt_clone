import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model import ThreeBlue1BrownNeuralNet
from utils import sigmoid, sigmoid_derivative

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

net = ThreeBlue1BrownNeuralNet()

epochs = 3
learning_rate = 0.01

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images_np = images.numpy()
        labels_np = labels.numpy()

        for i in range(images_np.shape[0]):
            image = images_np[i].flatten()
            label = labels_np[i]

            h1_pre = np.dot(net.w1, image) + net.b1
            h1 = sigmoid(h1_pre)
            
            h2_pre = np.dot(net.w2, h1) + net.b2
            h2 = sigmoid(h2_pre)
            
            output_pre = np.dot(net.w_out, h2) + net.b_out
            output = sigmoid(output_pre)

            target = np.zeros(10)
            target[label] = 1
            loss = np.sum((output - target) ** 2)
            running_loss += loss

            d_loss_d_output = 2 * (output - target)
            d_output_d_pre = sigmoid_derivative(output_pre)
            d_loss_d_pre_out = d_loss_d_output * d_output_d_pre

            d_loss_d_w_out = np.outer(d_loss_d_pre_out, h2)
            d_loss_d_b_out = d_loss_d_pre_out

            d_loss_d_h2 = np.dot(net.w_out.T, d_loss_d_pre_out)
            d_h2_d_pre = sigmoid_derivative(h2_pre)
            d_loss_d_pre_h2 = d_loss_d_h2 * d_h2_d_pre

            d_loss_d_w2 = np.outer(d_loss_d_pre_h2, h1)
            d_loss_d_b2 = d_loss_d_pre_h2

            d_loss_d_h1 = np.dot(net.w2.T, d_loss_d_pre_h2)
            d_h1_d_pre = sigmoid_derivative(h1_pre)
            d_loss_d_pre_h1 = d_loss_d_h1 * d_h1_d_pre
            
            d_loss_d_w1 = np.outer(d_loss_d_pre_h1, image)
            d_loss_d_b1 = d_loss_d_pre_h1

            net.w_out -= learning_rate * d_loss_d_w_out
            net.b_out -= learning_rate * d_loss_d_b_out
            net.w2 -= learning_rate * d_loss_d_w2
            net.b2 -= learning_rate * d_loss_d_b2
            net.w1 -= learning_rate * d_loss_d_w1
            net.b1 -= learning_rate * d_loss_d_b1

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainset)}")

print("Finished Training")

import pickle

model_params = {
    'w1': net.w1,
    'b1': net.b1,
    'w2': net.w2,
    'b2': net.b2,
    'w_out': net.w_out,
    'b_out': net.b_out
}

with open('../models/mnist_net.pkl', 'wb') as f:
    pickle.dump(model_params, f)

print("Model parameters saved to ../models/mnist_net.pkl")
