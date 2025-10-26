import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_loaders(
    batch_size: int = 4,
) -> tuple[DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = get_data_loaders(batch_size=8)
    for images, labels in trainloader:
        print(images.shape, labels.shape)
        break
