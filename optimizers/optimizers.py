import torch.optim as optim

def get_optimizer(optimizer_name, parameters, lr, momentum=0.9):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr)
    else:
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
    return optimizer