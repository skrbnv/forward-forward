import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class FFBlock(nn.Module):
    def __init__(self, ins, outs, optimizer=None, device='cpu', usenorm=False, useactivation=True, num_classes=10, threshold=2., name=''):
        super(FFBlock, self).__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.empty(1, ins))
        nn.init.kaiming_normal_(self.weights)
        self.layer = nn.Linear(ins, outs, bias=True)
        self.norm = nn.LayerNorm((ins)) if usenorm else nn.Identity()
        self.act = nn.ReLU() if useactivation is True else nn.Identity()
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.parameters(), lr=1e-4)
        self.to(device)

    def loss(self, inputs, statuses):
        ss = statuses.unsqueeze(-1)
        #  neg = (1-st)/2*inputs*torch.tanh(inputs)
        #  pos = (1+st)/2*torch.exp(inputs**2*-1)
        loss = (1-ss)/2 + ss*torch.exp(-1*inputs**2)
        return loss.mean()

    def square_loss(self, inputs, statuses):
        out = torch.sum(inputs**2, dim=-1)
        loss = statuses*(self.threshold-out)
        loss = torch.log(1. + torch.exp(loss)).mean()
        return loss

    def forward(self, x):
        x = self.norm(x)
        x = self.layer(x)
        x = self.act(x)
        return x

    def nograd_compute(self, inputs, labels):
        with torch.no_grad():
            output = self.compute(inputs, labels)
        return output

    def merge(self, x, y):
        x[:, :self.num_classes] = one_hot(y.flatten(), num_classes=self.num_classes)
        return x

    def compute(self, inputs, labels):
        return self.forward(self.merge(inputs, labels))

    def update(self, inputs, labels, statuses):
        # print(f'Updating layer {self.name}')
        y = self.forward(self.merge(inputs, labels))
        # loss = self.loss(y, statuses)
        loss = self.square_loss(y, statuses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            y = self.forward(self.merge(inputs, labels))
        return y, loss


class Model():
    def __init__(self, device='cpu', inner_sizes: list = [28 * 28, 100, 100, 10], num_classes: int = 10) -> None:
        self.device = device
        self.layers = []
        for i in range(len(inner_sizes) - 1):
            self.layers.append(
                FFBlock(ins=inner_sizes[i],
                        outs=inner_sizes[i + 1],
                        device=device,
                        usenorm=True if i > 0 else False,
                        useactivation=True if i < len(inner_sizes)-2 else False,
                        num_classes=num_classes,
                        name=f"L{i}"))

    def update(self, inputs, labels, statuses):
        x = inputs
        losses = []
        for i, layer in enumerate(self.layers):
            x, loss = layer.update(x, labels, statuses)
            losses += [loss.item()]
        return x, losses

    def update_layer(self, inputs, labels, statuses, layer_num):
        x = inputs
        losses = []
        for i, layer in enumerate(self.layers):
            if i == layer_num:
                x, loss = layer.update(x, labels, statuses)
                losses += [loss.item()]
            else:
                x = layer.nograd_compute(x, labels)
        return x, losses

    def compute(self, inputs, labels):
        x = inputs
        with torch.no_grad():
            for layer in self.layers:
                x = layer.compute(x, labels)
        return x

    def goodness(self, inputs, labels):
        x = self.compute(inputs, labels)
        return (x**2).sum(dim=1)

    def layer_count(self):
        return len(self.layers)
