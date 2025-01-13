import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.helper import load_image, class_names
import numpy as np
from skimage.segmentation import slic
from visualizations.visualizer import visualize_superpixels
from utils.gradcam import GradCAM, overlay_heatmap
import torch.nn.functional as F
import cv2
import matplotlib as plt
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
    
    visualize_superpixels(images_to_explain, segments_slic)


    for i in range(len(images_to_explain)):
      
        with torch.no_grad():
            output = net(images_to_explain[i])
            _, predicted = torch.max(output, 1)
            print(f'Image {i}: Predicted class index: {predicted.item()}, Predicted class name: {class_names[predicted.item()]}')

        heatmap = grad_cam.generate_heatmap(predicted.item())
        overlay = overlay_heatmap(image_paths[i], heatmap)


        plt.imshow(overlay)
        plt.title(f"Prediction: {class_names[predicted.item()]} ")
        plt.axis("off")
        plt.show()