import streamlit as st
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from models.resnet import ResNet
from utils.helper import load_image, class_names
from optimizers.optimizers import get_optimizer
from transforms.transforms import transform_train, transform_test
from trainer.trainer import train_model, evaluate_model
import os
import numpy as np
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'best_resnet_model.pth'


@st.cache_resource
def load_model():
    net = ResNet(num_classes=10).to(device)
    if os.path.exists(MODEL_SAVE_PATH):
         print("Loading pre-trained model...")
         net.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Training model from scratch...")
        train_data = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_data = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer('SGD', net.parameters(), 0.002155517765181785)
        net = train_model(net, train_loader, optimizer, criterion, num_epochs=30, device=device)
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
    return net


def predict_and_explain(image, net, image_path):
    net.eval()
    output = net(image)
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()
    
    st.write(f'Predicted class index: {class_idx}, Predicted class name: {class_names[class_idx]}')

    net_for_gradcam = ResNet(num_classes=10).to(device)
    net_for_gradcam.load_state_dict(net.state_dict())

    target_layer = net_for_gradcam.layer4[-1]
    grad_cam = utils.gradcam.GradCAM(net_for_gradcam, target_layer)
    heatmap = grad_cam.generate_heatmap(image, class_idx) 


    original_image = Image.open(image_path)
    original_image = np.array(original_image)
    overlay = utils.gradcam.overlay_heatmap(original_image, heatmap)

    if overlay is not None:
        overlay_image = Image.fromarray(overlay) 
        st.image(overlay_image, caption='Grad-CAM Heatmap', use_column_width=True)
    else:
      st.write("Could not generate Grad-CAM Heatmap")

def main():
    st.title("Image Classifier with Grad-CAM")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    net = load_model()
    if uploaded_file is not None:
        try:
          
          image_path = uploaded_file.name  
          image_pil = Image.open(uploaded_file)
          
          
          image = utils.helper.new_transform(image_pil)
          image = image.unsqueeze(0).to(device)
          predict_and_explain(image, net, image_path)
        except Exception as e:
           st.error(f"Error processing image: {e}")
           
if __name__ == "__main__":
    import utils.gradcam, utils.helper
    main()