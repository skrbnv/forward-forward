from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torch.utils.data import default_collate
from torchvision import transforms
from typing import Tuple, Any
import torch
from random import choice, random


class MNIST_Collate:
    def __init__(self, num_classes=10, device="cpu") -> None:
        self.num_classes = num_classes
        self.device = device
        self.classlist = [el for el in range(num_classes)]

    def __call__(self, batch) -> Any:
        output = []
        pos_state = torch.tensor(1, dtype=torch.long, device=self.device)
        neg_state = torch.tensor(-1, dtype=torch.long, device=self.device)
        for img, pos_label in batch:
            if random() >= 0.5:
                classlist = self.classlist.copy()
                del classlist[pos_label.item()]
                neg_label = torch.tensor(
                    choice(classlist), dtype=torch.long, device=self.device
                )
                output += [(img, neg_label, neg_state)]
            else:
                output += [(img, pos_label, pos_state)]
        return default_collate(output)


class MNIST_GPU(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(root, train, download)
        tfn = transforms.Normalize((0.1307,), (0.3081,))
        self.data = tfn((self.data / 255.0).float().unsqueeze(1).to(device))
        self.targets = self.targets.long().to(device)
        self.augment = transforms.Compose(
            [
                transforms.Pad(2, fill=0, padding_mode="constant"),
                transforms.RandomCrop(28),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.data[index], self.targets[index]
        if self.train is True:
            img = self.augment(img)
        return (img, label)


def generate(batch_size: list = [16, 16], num_workers: list = [0, 0], device="cpu"):
    trainset = MNIST_GPU(root="./MNIST/train", train=True, download=True, device=device)
    testset = MNIST_GPU(root="./MNIST/test", train=False, download=True, device=device)
    traintestset = Subset(
        MNIST_GPU(root="./MNIST/train", train=True, download=True, device=device),
        range(10000),
    )
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
        collate_fn=MNIST_Collate(device=device),
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
    )
    train_test_loader = DataLoader(
        traintestset,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers[0],
    )
    return train_loader, test_loader, train_test_loader
