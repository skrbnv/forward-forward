import torch
from tqdm import tqdm


def test_loop(loader, model, num_classes, device):
    correct, total = 0, 0
    for inputs, labels, _ in (pbar := tqdm(loader)):
        for i in range(inputs.size(0)):
            with torch.no_grad():
                x = model.goodness(
                    inputs[i].unsqueeze(0).repeat((num_classes, 1, 1, 1)),
                    torch.tensor([el for el in range(num_classes)]).to(device),
                )
            correct += 1 if (torch.argmax(x) == labels[i]).item() is True else 0
            total += 1
            pbar.set_description_str(
                f"├── C: {correct}, T: {total}, Acc: {correct*100/total:.2f}%"
            )
    return correct * 100 / total
