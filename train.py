from libs.classes import Model
from libs.collate import Collate_Train, Collate_Test
import libs.utils as _utils
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import os


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
CONFIG = _utils.load_yaml()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(
    device=device, inner_sizes=CONFIG.dimensions, num_classes=CONFIG.num_classes
)
trainset = MNIST(root="./MNIST/train", train=True, download=True, transform=None)
testset = MNIST(root="./MNIST/test", train=False, download=True, transform=None)
train_loader = DataLoader(
    trainset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=Collate_Train(device=device),
)
test_loader = DataLoader(
    testset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=Collate_Test(device=device),
)


for epoch in range(CONFIG.num_epochs):
    print(f"├──────────── CYCLE {epoch + 1}/{CONFIG.num_epochs} ────────────")
    for i in range(model.layer_count()):
        print(f"├ Optimizing layer {i+1}/{model.layer_count()}")
        for _pass in range(CONFIG["num_passes"]):
            # train
            loss = []
            zeros = 0
            for inputs, labels, statuses in (pbar := tqdm(train_loader)):
                x, losses = model.update_layer(inputs, labels, statuses, i)
                if torch.max(x) == 0:
                    zeros += 1
                loss += losses
                pbar.set_description(
                    f"├── [{zeros}] Pass {_pass+1}/{CONFIG.num_passes}: {torch.mean(torch.tensor(loss)).item():.4f}"
                )

    # test
    correct, total, zeros = 0, 0, 0
    for inputs, labels, _ in (pbar := tqdm(test_loader)):
        for i in range(inputs.size(0)):
            x = model.goodness(
                inputs[i].unsqueeze(0).repeat((CONFIG.num_classes, 1)),
                torch.tensor([[el] for el in range(CONFIG.num_classes)]).to(device),
            )
            if x.max() == 0:
                zeros += 1
            correct += 1 if (torch.argmax(x) == labels[i]).item() is True else 0
            total += 1
            pbar.set_description_str(
                f"├── [{zeros}] C: {correct}, T: {total}, Acc: {correct*100/total:.2f}%"
            )
