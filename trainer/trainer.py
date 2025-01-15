import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.helper import load_image, class_names
import numpy as np
from visualizations.visualizer import visualize_superpixels
from utils.gradcam import GradCAM, overlay_heatmap
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def train_model(net, train_loader, optimizer, criterion, num_epochs, device):
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return net

def evaluate_model(net, test_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2f}')
    return accuracy

def calculate_and_visualize_explanations(net, image_paths, class_names, device, segments_slic):
    net.eval()
    target_layer = net.layer4[-1]
    grad_cam = GradCAM(net, target_layer)
    images_to_explain = [load_image(img,device) for img in image_paths]

    for i, image in enumerate(images_to_explain):
        output = net(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        print(f'Image {i}: Predicted class index: {class_idx}, Predicted class name: {class_names[class_idx]}')

        net.zero_grad()
        output[0, class_idx].backward()
        heatmap = grad_cam.generate_heatmap(image, class_idx)
        
        # Load the image using PIL
        original_image = Image.open(image_paths[i])
        original_image = np.array(original_image)
      
        overlay = overlay_heatmap(original_image, heatmap)

        if overlay is not None:
            plt.imshow(overlay)
            plt.title(f"Prediction: {class_names[class_idx]} ")
            plt.axis("off")
            plt.show()