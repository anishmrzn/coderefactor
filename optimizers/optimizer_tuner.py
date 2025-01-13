import optuna
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torch

from models.resnet import ResNet 
from optimizers.optimizers import get_optimizer 
from transforms.transforms import transform_train, transform_test  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def objective(trial, train_data, test_data):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = ResNet(num_classes=10).to(device)
    
    optimizer = get_optimizer(optimizer_name, net.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        net.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
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
    return accuracy

def tune_hyperparameters(train_data, test_data, n_trials=20):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, train_data, test_data), n_trials=n_trials)
    best_params = study.best_params
    best_value = study.best_value
    return best_params, best_value