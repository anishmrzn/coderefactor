import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
      self.gradients = grad_output[0]

    def generate_heatmap(self, image, class_idx):
        self.gradients = None
        self.feature_maps = None
        
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

        self.model.zero_grad()
        output = self.model(image)
        output[0, class_idx].backward()

        if self.gradients is None:
            print("Gradients are None")
            return None
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().detach().numpy()


def overlay_heatmap(original_image, heatmap):
    if heatmap is None:
        print("Heatmap is None")
        return None

    if original_image is None:
        print(f"Error: Could not load image.")
        return None
      
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_image = heatmap_image.convert("RGB").resize(original_image.shape[1::-1], Image.Resampling.BILINEAR)


    original_image = Image.fromarray(original_image).convert("RGB")

    overlay = Image.blend(original_image, heatmap_image, 0.5)
    return np.array(overlay)