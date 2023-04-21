import torch
import torch.nn as nn
from torch.nn.functional import one_hot


def is_ff(obj):
    return True if isinstance(obj, FFBlock) or isinstance(obj, FFConvBlock) else False


class FFBlock(nn.Module):
    def __init__(
        self,
        ins,
        outs,
        name,
        optimizer=None,
        device="cpu",
        usenorm=False,
        useactivation=True,
        num_classes=10,
        threshold=2.0,
    ):
        super(FFBlock, self).__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.num_classes = num_classes
        self.layer = nn.Linear(ins, outs, bias=True)
        self.norm = nn.LayerNorm((ins)) if usenorm else nn.Identity()
        self.act = nn.ReLU() if useactivation is True else nn.Identity()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(self.parameters(), lr=1e-4)
        )
        self.to(device)

    def loss(self, inputs, statuses):
        out = torch.sum(inputs**2, dim=-1)
        loss = statuses * (self.threshold - out)
        loss = torch.log(1.0 + torch.exp(loss)).mean()
        return loss

    def forward(self, x):
        x = self.norm(x)
        x = self.layer(x)
        x = self.act(x)
        return x

    def merge(self, x, y):
        x[:, : self.num_classes] = one_hot(y.flatten(), num_classes=self.num_classes)
        return x

    def compute(self, inputs, labels):
        return self.forward(self.merge(inputs, labels))

    def update(self, inputs, labels, statuses):
        # print(f'Updating layer {self.name}')
        y = self.forward(self.merge(inputs, labels))
        # loss = self.loss(y, statuses)
        loss = self.loss(y, statuses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            y = self.forward(self.merge(inputs, labels))
        return y, loss


class FFConvBlock(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size,
        stride,
        padding,
        name,
        optimizer=None,
        device="cpu",
        usenorm=False,
        useactivation=True,
        num_classes=10,
        threshold=2.0,
    ):
        super().__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.num_classes = num_classes
        self.layer = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(channels_in) if usenorm else nn.Identity()
        self.act = nn.ReLU() if useactivation is True else nn.Identity()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(self.parameters(), lr=1e-4)
        )
        self.to(device)

    def loss(self, inputs, statuses):
        out = torch.sum(inputs**2, dim=(-2, -1)).mean(-1)
        #  out = torch.max(torch.sum(inputs**2, dim=(-2, -1)), dim=-1).values
        loss = statuses * (self.threshold - out)
        loss = torch.log(1.0 + torch.exp(loss)).mean()
        return loss

    def tahn_loss(self, inputs, statuses):
        out = torch.sum(inputs**2, dim=(-2, -1)).mean(-1)
        losses = out - statuses * self.threshold
        return losses.mean()

    def forward(self, x):
        x = self.norm(x)
        x = self.layer(x)
        x = self.act(x)
        return x

    def merge(self, x, y):
        x[:, :, 0, : self.num_classes] = (
            one_hot(y.flatten(), num_classes=self.num_classes)
            .unsqueeze(1)
            .repeat(1, x.size(1), 1)
        )
        return x

    def compute(self, inputs, labels):
        return self.forward(self.merge(inputs, labels))

    def update(self, inputs, labels, statuses):
        # print(f'Updating layer {self.name}')
        y = self.forward(self.merge(inputs, labels))
        loss = self.loss(y, statuses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            y = self.forward(self.merge(inputs, labels))
        return y, loss
