from models.conv import Model
from libs.test_loop import test_loop
from libs.ffmodel import is_ff
import libs.utils as _utils
import libs.dataset as dataset
import torch
from tqdm import tqdm
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False, help="sync with W&B")
args = parser.parse_args()
WANDB = args.wandb
CONFIG = _utils.load_yaml()
if WANDB:
    wprj = wandb.init(project=CONFIG.wandb.project, resume=False, config=CONFIG)
    RUN_ID = wprj.id
else:
    RUN_ID = _utils.get_random_hash()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device=device, num_classes=CONFIG.num_classes)
print(model)

train_loader, test_loader, train_test_loader = dataset.generate(
    CONFIG.dataset.batch_size, CONFIG.dataset.num_workers, device
)


for epoch in range(CONFIG.num_cycles):
    print(f"├──────────── CYCLE {epoch + 1}/{CONFIG.num_cycles} ────────────")
    for i in range(model.layer_count()):
        if not is_ff(model.layers[i]):
            print(
                f"├ ..skipping layer {i+1}/{model.layer_count()}:"
                f" {model.layers[i].__class__.__name__}"
            )
            continue
        print(
            f"├ Optimizing layer {i+1}/{model.layer_count()}:"
            f" {model.layers[i].__class__.__name__} ({model.layers[i].name})"
        )
        # train
        for _pass in range(CONFIG.num_epochs):
            losses = []
            for inputs, labels, states in (pbar := tqdm(train_loader)):
                # inputs = inputs.flatten(1)  # add for linear
                x, loss = model.update_layer(inputs, labels, states, i)
                losses += loss
                pbar.set_description(
                    f"├── [{_pass+1}/{CONFIG.num_epochs}]:"
                    f" {torch.mean(torch.tensor(losses)).item():.4f}"
                )

    # test
    train_acc = test_loop(train_test_loader, model, CONFIG.num_classes, device)
    test_acc = test_loop(test_loader, model, CONFIG.num_classes, device)

    if WANDB:
        wandb.log({"Training acc": train_acc, "Test acc": test_acc})

    _utils.checkpoint(
        id=RUN_ID, data={"epoch": epoch, "state_dict": model.state_dict()}
    )
