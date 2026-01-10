import importlib
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        target = image.clone()
        # onehot labels
        labels = np.zeros(10)
        labels[label] = 1
        
        return image, target, torch.tensor(labels, dtype=torch.float32)
    

def get_mnist(path: str = '../data/') -> DataLoader | DataLoader:
    """Get the MNIST training and testing data loaders.

    Args:
        batch_size (int, optional): Defaults to 32.

    Returns:
        Dataset | Dataset: Training and testing datasets
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=path, train=False, download=True, transform=transform)

    train_noisy_dataset = MNISTDataset(train_dataset)
    test_noisy_dataset = MNISTDataset(test_dataset)

    return train_noisy_dataset, test_noisy_dataset


def add_noise(x, noise_factor):
    noisy_x =  (1-noise_factor)*x + noise_factor * torch.randn(x.size())
    return noisy_x

def get_device(devices: List[str] = ["xpu", "cuda", "mps", "cpu"]):
    """
    Get the first available device from devices list.
    """
    for device in devices:
        try:
            device_module = importlib.import_module(f"torch.{device}")
        except ModuleNotFoundError:
            device_module = None
        if getattr(device_module, "is_available", lambda: False)():
            break

    return device

def get_device_type(device: Union[str, torch.device]):
    """
    Get string indicating device type of device.
    """
    return (device.type if isinstance(device, torch.device) else device)

def get_backend(device: Union[str, torch.device], backends: Dict[str, str] = {
    "cpu": "gloo", "cuda": "nccl", "mps": "", "xpu": "xccl"}):
    """
    For specified device, get distributed-processing backend.
    """
    return backends.get(get_device_type(device), None)
