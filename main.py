import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import optuna
from models.resnet import ResNet
from utils.helper import load_image, class_names, create_segments
from optimizers.optimizers import get_optimizer
from optimizers.optimizer_tuner import tune_hyperparameters
from transforms.transforms import transform_train, transform_test
from trainer.trainer import train_model, evaluate_model, calculate_and_visualize_explanations
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    net.load_state_dict(torch.load('best_resnet_model.pth'))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    net = ResNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, net.parameters(), lr)
        
    net = train_model(net, train_loader, optimizer, criterion, num_epochs=30, device=device)
    torch.save(net.state_dict(), 'best_resnet_model.pth')
    
    evaluate_model(net, test_loader, device=device)

    net = ResNet(num_classes=10).to(device)
    net.load_state_dict(torch.load('best_resnet_model.pth'))
    image_paths = ['example2.jpg','example.jpg','example3.jpg']
    images_to_explain = [load_image(img,device) for img in image_paths]
    
    segments_slic = [create_segments(image.squeeze().cpu().numpy().transpose(1, 2, 0)) for image in images_to_explain]
    
    for image_path, image in zip(image_paths, images_to_explain): 
      
        outputs = net(image)
        _, predicted = torch.max(outputs, 1) 
        class_idx = predicted.item() 
        net.zero_grad() 
        outputs[0, class_idx].backward()

    calculate_and_visualize_explanations(net, image_paths, class_names, x_test, device, segments_slic=segments_slic)