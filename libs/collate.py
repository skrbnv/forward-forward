import torch
import random
from torch.utils.data import default_collate
from typing import Any


class Collate_Train:
    def __init__(self, num_classes=10, device="cpu") -> None:
        self.num_classes = num_classes
        self.device = device

    def __call__(self, batch) -> Any:
        # batch is a list of tuples, each tuple is a torch.tensor (1,28,28) and int
        output = []
        for tpl in batch:
            status = 1
            image, label = tpl
            true_label = label
            if random.random() >= 0.5:
                label = random.choice(
                    [el for el in range(self.num_classes) if el != label]
                )
                status = -1
            label = torch.tensor([label], dtype=torch.long)
            assert (status == 1 and label == true_label) or (
                status == -1 and label != true_label
            ), "Label generator broken!"
            output += [
                (
                    image.to(self.device),
                    label.to(self.device),
                    torch.tensor(status).to(self.device),
                )
            ]
        return default_collate(output)


class Collate_Test:
    def __init__(self, num_classes=10, device="cpu") -> None:
        self.num_classes = num_classes
        self.device = device

    def __call__(self, batch) -> Any:
        output = []
        for tpl in batch:
            image, label = tpl
            label = torch.tensor([label], dtype=torch.long)
            output += [(image.to(self.device), label.to(self.device), 1)]
        return default_collate(output)
