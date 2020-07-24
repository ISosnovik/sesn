'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn


def train_xent(model, optimizer, loader, device=torch.device('cuda')):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_acc(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy
