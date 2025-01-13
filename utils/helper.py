import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.segmentation import slic

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

new_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def load_image(image_path, device):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0).to(device)  
    return image
    
def create_segments(image, n_segments=100, compactness=10):

    original_image = (image + 1) / 2 * 255
    original_image = original_image.astype(np.uint8)
    return slic(original_image, n_segments=n_segments, compactness=compactness)