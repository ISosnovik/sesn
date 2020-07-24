'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import json

from .cutout import Cutout


mean = {
    'stl10': (0.4467, 0.4398, 0.4066),
    'scale_mnist': (0.0607,),
}

std = {
    'stl10': (0.2603, 0.2566, 0.2713),
    'scale_mnist': (0.2161,),
}


def loader_repr(loader):
    # fix some problems with torchvision 0.3.0 with VisionDataset
    if not isinstance(loader.dataset, ConcatDataset):
        s = ('{dataset.__class__.__name__} Loader: '
             'num_workers={num_workers}, '
             'pin_memory={pin_memory}, '
             'sampler={sampler.__class__.__name__}\n'
             'Root: {dataset.root}\n'
             )
        s = s.format(**loader.__dict__)
        s += 'Data Points: {}\n{}\nTransforms:\n{}'
        s = s.format(len(loader.dataset), loader.dataset.extra_repr(), loader.dataset.transform)
        return s
    else:
        s = ('{dataset.__class__.__name__} Loader: '
             'num_workers={num_workers}, '
             'pin_memory={pin_memory}, '
             'sampler={sampler.__class__.__name__}\n'
             )
        s = s.format(**loader.__dict__)
        for d in loader.dataset.datasets:
            s += '| Dataset {}, Root: {}\n'.format(d.__class__.__name__, d.root)
        s += 'Data Points: {}\n \nTransforms:' + '\n{}' * len(loader.dataset.datasets)
        s = s.format(len(loader.dataset), *[d.transform for d in loader.dataset.datasets])
        return s


#################################################
##################### STL-10 ####################
#################################################
def stl10_plus_train_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10']),
        Cutout(1, 32),
    ])
    dataset = datasets.STL10(root=root, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def stl10_test_loader(batch_size, root, download=True):
    transform = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean['stl10'], std['stl10'])
    ])
    dataset = datasets.STL10(root=root, split='test', transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader


#################################################
##################### SCALE #####################
#################################################
def scale_mnist_train_loader(batch_size, root, extra_scaling=1):
    transform_modules = []
    if not extra_scaling == 1:
        if extra_scaling > 1:
            extra_scaling = 1 / extra_scaling
        scale = (extra_scaling, 1 / extra_scaling)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        transform_modules.append(scaling)

    transform_modules = transform_modules + [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ]

    transform = transforms.Compose(transform_modules)
    dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def scale_mnist_val_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=2)
    return loader


def scale_mnist_test_loader(batch_size, root):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist'])
    ])
    dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=2)
    return loader
