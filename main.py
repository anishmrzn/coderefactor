import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from models.resnet import ResNet
from utils.helper import load_image, class_names, create_segments
from optimizers.optimizers import get_optimizer
from transforms.transforms import transform_train, transform_test
from trainer.trainer import train_model, evaluate_model, calculate_and_visualize_explanations
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'best_resnet_model.pth'


if __name__ == '__main__':
    train_data = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_data = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    
    test_loader_all = DataLoader(test_data, batch_size=len(test_data), shuffle=False) 
    x_test, y_test = next(iter(test_loader_all))
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    lr = 0.002155517765181785
    batch_size = 16
    optimizer_name = 'SGD'

    print("Using best parameters found from previous trails: ", {'lr': lr, 'batch_size': batch_size, 'optimizer': optimizer_name})

    net = ResNet(num_classes=10).to(device)

    # Load model if it exists, otherwise train it
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Training model from scratch...")
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(optimizer_name, net.parameters(), lr)
        net = train_model(net, train_loader, optimizer, criterion, num_epochs=30, device=device)
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
        evaluate_model(net, test_loader, device=device)

    image_paths = ['example2.jpg','example.jpg','example3.jpg']
    images_to_explain = [load_image(img,device) for img in image_paths]

    segments_slic = [create_segments(image.squeeze().cpu().numpy().transpose(1, 2, 0)) for image in images_to_explain]
    
    calculate_and_visualize_explanations(net, image_paths, class_names, device, segments_slic)