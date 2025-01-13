import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np

def visualize_superpixels(images, segments_slic):
    for i, (image, segments) in enumerate(zip(images, segments_slic)):
        plt.imshow(mark_boundaries(image.squeeze().cpu().numpy().transpose(1, 2, 0) / 2 + 0.5, segments))
        plt.title(f"Superpixels for Image {i}")
        plt.show()