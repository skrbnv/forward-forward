import torch
import torch.nn as nn
from torch.nn.functional import one_hot


def is_ff(obj):
    return True if isinstance(obj, FFBlock) else False


class FFBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = None
        self.act = None
        self.optimizer = None
        self.activation = None

    def norm(self, x):
        raise Exception("Not implemented")

    def loss(self):
        raise Exception("Not Implemented")

    def forward(self, x, labels):
        x = self.merge(x, labels)
        x = self.norm(x)
        x = self.layer(x)
        x = self.act(x)
        return x

    def merge(self):
        raise Exception("Not Implemented")

    def update(self, inputs, labels, statuses):
        # print(f'Updating layer {self.name}')
        y = self.forward(inputs, labels)
        loss = self.loss(y, statuses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            y = self.forward(inputs, labels)
        return y, loss


class FFConvBlock(FFBlock):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size,
        stride,
        padding,
        name="FFConvBlock",
        optimizer=None,
        device="cpu",
        activation=nn.ReLU(),
        num_classes=10,
        threshold=2.0,
    ):
        super().__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.norm_fn = nn.BatchNorm2d(channels_in)
        if name == "conv1":
            self.norm_fn = nn.Identity()
        self.num_classes = num_classes
        self.layer = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
        # self.norm = nn.BatchNorm2d(channels_in) if use_norm else nn.Identity()
        self.act = activation
        self.optimizer = (
            torch.optim.SGD(self.parameters(), lr=1e-4)
            if optimizer is None
            else optimizer
        )
        self.to(device)

    def norm(self, x):
        # dim = (2, 3)
        # return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + 1e-5)
        return self.norm_fn(x)

    def loss(self, inputs, statuses):
        out = torch.sum(inputs**2, dim=(-2, -1)).mean(-1)
        loss = statuses * (self.threshold - out)
        loss = torch.log(1.0 + torch.exp(loss)).mean()
        return loss

        # norm
        """
        elif len(x.shape) == 4:
            dim = (2, 3)
        else:
            raise Exception("Incorrect data shape")
        return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + 1e-5)
        """

    def merge(self, x, y):
        shape = x.shape
        x = x.flatten(-2)
        x[:, :, : self.num_classes] = (
            one_hot(y, num_classes=self.num_classes)
            .unsqueeze(1)
            .repeat(1, x.size(1), 1)
        )
        return x.reshape(shape)


class FFLinearBlock(FFBlock):
    def __init__(
        self,
        ins,
        outs,
        name="FFLinearBlock",
        optimizer=None,
        device="cpu",
        activation=nn.ReLU(),
        num_classes=10,
        threshold=2.0,
    ):
        super().__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.norm_fn = nn.LayerNorm((ins))
        self.num_classes = num_classes
        self.layer = nn.Linear(ins, outs, bias=True)
        # self.norm = nn.LayerNorm((ins)) if use_norm else nn.Identity()
        self.act = activation
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(self.parameters(), lr=1e-4)
        )
        self.to(device)

    def norm(self, x):
        return self.norm_fn(x)

    def loss(self, inputs, statuses):
        out = torch.sum(inputs**2, dim=-1)
        loss = statuses * (self.threshold - out)
        loss = torch.log(1.0 + torch.exp(loss)).mean()
        return loss

        # return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + 1e-5)

    def merge(self, x, y):
        x[:, : self.num_classes] = one_hot(y.flatten(), num_classes=self.num_classes)
        return x
