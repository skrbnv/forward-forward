import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Optional


def is_ff(obj):
    return True if isinstance(obj, FFBlock) else False


def metric(x):
    if len(x.shape) == 4:
        output = x.pow(2).sum(dim=(2, 3)).sqrt().mean(dim=1)
    elif len(x.shape) == 2:
        output = x.pow(2).sum(dim=1).sqrt()
    else:
        raise Exception("Unsupported shape")
    return output


class SimpleNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            return x / (torch.linalg.norm(x, ord=2, dim=1, keepdim=True) + 1e-4)
        elif len(x.shape) == 4:
            return x / (torch.linalg.norm(x, ord=2, dim=(-1, -2), keepdim=True) + 1e-4)
        else:
            raise Exception("Unsupported shape")


class FFBLockAfterInit(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.after_init()
        return instance


class FFBlock(nn.Module):
    def __init__(
        self,
        name: Optional[str] = "FFBlock",
        threshold: Optional[float] = 1.6,
        spacing: Optional[float] = 0,
        num_classes: Optional[int] = 10,
        activation: Optional[nn.Module] = nn.ReLU(),
        optimizer: Optional[nn.Module] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.spacing = torch.tensor(spacing).to(device)
        self.num_classes = num_classes
        self.norm_fn = None
        self.layer = None
        self.act = activation
        self.optimizer_var = optimizer
        self.device = device

    def after_init(self):
        self.optimizer = (
            torch.optim.SGD(self.parameters(), lr=5e-3)
            if self.optimizer_var is None
            else self.optimizer_var
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 20], gamma=0.5
        )
        self.to(self.device)

    def norm(self, x):
        if self.norm_fn is None or isinstance(self.norm_fn, nn.Identity):
            return x
        else:
            return self.norm_fn(x)

    def loss(self):
        raise NotImplementedError

    def forward(self, x, labels):
        x = self.merge(x, labels)
        x = self.norm(x)
        x = self.layer(x)
        x = self.act(x)
        return x

    def merge(self):
        raise NotImplementedError

    def update(self, inputs, labels, states):
        # print(f'Updating layer {self.name}')
        y = self.forward(inputs, labels)
        loss = self.loss(y, states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            y = self.forward(inputs, labels)
        return y, loss


# #######################################################
#                      Conv2d block
# #######################################################
class FFConvBlock(FFBlock, metaclass=FFBLockAfterInit):
    def __init__(
        self,
        channels_in,
        channels_out,
        shape_in,
        kernel_size,
        stride,
        padding,
        device="cpu",
        norm=nn.Identity(),
        *args,
        **kwargs,
    ):
        super().__init__(device=device, *args, **kwargs)
        self.norm_fn = norm
        self.layer = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)

    def loss(self, inputs, states):
        subs = states * (self.threshold - metric(inputs))
        losses = torch.log(1 + torch.cosh(subs + self.spacing) + subs + self.spacing)
        # losses = torch.sigmoid(subs)
        return losses.mean()

    def merge(self, x, y):
        shape = x.shape
        x = x.flatten(-2)
        x[:, :, : self.num_classes] = (
            one_hot(y, num_classes=self.num_classes)
            .unsqueeze(1)
            .repeat(1, x.size(1), 1)
        )
        return x.reshape(shape)


# #######################################################
#                      Linear block
# #######################################################
class FFLinearBlock(FFBlock, metaclass=FFBLockAfterInit):
    def __init__(
        self,
        ins: int,
        outs: int,
        device: Optional[str] = "cpu",
        norm=None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, *args, **kwargs)
        self.norm_fn = norm if norm is not None else nn.LayerNorm((ins))
        self.layer = nn.Linear(ins, outs, bias=True)
        self.to(device)

    def loss(self, inputs, states):
        subs = states * (self.threshold - metric(inputs))
        losses = torch.log(1 + torch.cosh(subs + self.spacing) + subs + self.spacing)
        # losses = torch.sigmoid(subs)
        return losses.mean()

    def merge(self, x, y):
        x[:, : self.num_classes] = one_hot(y.flatten(), num_classes=self.num_classes)
        return x
