import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

from model import ThreeBlue1BrownNeuralNet

with open('../models/mnist_net.pkl', 'rb') as f:
    model_params = pickle.load(f)

net = ThreeBlue1BrownNeuralNet()
net.w1 = model_params['w1']
net.b1 = model_params['b1']
net.w2 = model_params['w2']
net.b2 = model_params['b2']
net.w_out = model_params['w_out']
net.b_out = model_params['b_out']

print("Model parameters loaded successfully.")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

correct_count = 0
total_count = 0
for image, label in testloader:
    image_np = image.numpy().flatten()
    label_np = label.numpy()[0]


    output = net.feedforward(image_np)
    prediction = np.argmax(output)

    if prediction == label_np:
        correct_count += 1
    total_count += 1

accuracy = (correct_count / total_count) * 100
print(f"\nAccuracy on the test set: {accuracy:.2f}%")


print("\n--- Testing a random image ---")
random_idx = random.randint(0, len(testset) - 1)
image, label = testset[random_idx]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"True Label: {label}")
plt.show()

image_np = image.numpy().flatten()
output = net.feedforward(image_np)
prediction = np.argmax(output)

print(f"Model Prediction: {prediction}")
print(f"Output vector: \n{output}")
