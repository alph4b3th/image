import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

batch = 64
train = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
        )

test = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
        )

train_dataloader = DataLoader(train, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch, shuffle=True)
