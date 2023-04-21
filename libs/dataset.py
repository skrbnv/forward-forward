from libs.collate import Collate_Train, Collate_Test
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def generate(batch_size: list = [16, 16], num_workers: list = [0, 0], device="cpu"):
    transforms_train = transforms.Compose(
        [
            transforms.Pad(2, fill=0, padding_mode="constant"),
            transforms.RandomCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trainset = MNIST(
        root="./MNIST/train", train=True, download=True, transform=transforms_train
    )
    testset = MNIST(
        root="./MNIST/test", train=False, download=True, transform=transforms_test
    )
    traintestset = MNIST(
        root="./MNIST/train", train=True, download=True, transform=transforms_test
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
        collate_fn=Collate_Train(device=device),
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
        collate_fn=Collate_Test(device=device),
    )
    train_test_loader = DataLoader(
        traintestset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
        collate_fn=Collate_Test(device=device),
    )
    return train_loader, test_loader, train_test_loader
