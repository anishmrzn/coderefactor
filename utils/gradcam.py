import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_backward_hook(self.save_gradients)  

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, class_idx):
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().detach().numpy()

def overlay_heatmap(image_path, heatmap):
    original_image = cv2.imread(image_path)

    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_image.shape[-1] == 3:
       original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    return overlay