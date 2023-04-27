import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Optional


def is_ff(obj):
    return (
        True if isinstance(obj, FFBlock) or isinstance(obj, FFBolzmannChain) else False
    )


class FFBLockAfterInit(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.after_init()
        return instance


class FFBlock(nn.Module):
    def __init__(
        self,
        name: Optional[str] = "FFBlock",
        threshold: Optional[float] = 2.0,
        num_classes: Optional[int] = 10,
        activation: Optional[nn.Module] = nn.ReLU(),
        optimizer: Optional[nn.Module] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = name
        self.threshold = torch.tensor(threshold).to(device)
        self.num_classes = num_classes
        self.norm_fn = None
        self.layer = None
        self.act = activation
        self.optimizer_var = optimizer
        self.device = device

    def after_init(self):
        self.optimizer = (
            torch.optim.SGD(self.parameters(), lr=1e-2)
            if self.optimizer_var is None
            else self.optimizer_var
        )
        self.to(self.device)

    def norm(self, x):
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
        norm=None,
        *args,
        **kwargs,
    ):
        super().__init__(device=device, *args, **kwargs)
        self.norm_fn = norm if norm is not None else nn.LayerNorm(shape_in)
        # self.simple_norm
        self.layer = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)

    # def simple_norm(self, x):
    #    return x / (torch.linalg.norm(x, dim=(1, 2, 3), keepdim=True) + 1e-5)

    def loss(self, inputs, states):
        sqs = torch.sqrt(torch.sum(inputs**2, dim=(2, 3))).mean(dim=1)
        subs = states * (self.threshold - sqs)
        losses = torch.log(1 + torch.cosh(subs) + subs) / 2.0
        # losses = torch.log(1.0 + torch.exp(subs))
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
        self.norm_fn = (
            norm if norm is not None else nn.LayerNorm((ins))
        )  # self.simple_norm
        self.layer = nn.Linear(ins, outs, bias=True)
        self.to(device)

    # def simple_norm(self, x):
    #    return x / (torch.linalg.norm(x, dim=1, keepdim=True) + 1e-5)

    def loss(self, inputs, states):
        sqs = torch.sqrt(torch.sum(inputs**2, dim=1))
        subs = states * (self.threshold - sqs)
        losses = torch.log(1 + torch.cosh(subs) + subs) / 2.0
        # losses = torch.log(1.0 + torch.exp(subs))
        return losses.mean()

    def merge(self, x, y):
        x[:, : self.num_classes] = one_hot(y.flatten(), num_classes=self.num_classes)
        return x


class FFBolzmannChain(nn.Module):
    def __init__(
        self,
        bounces: int = 3,
        nodes: list = [],
        name: str = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        assert len(nodes) > 0, "Empty nodes list"
        self.bounces = bounces
        self.nodes = nodes
        self.name = name if name is not None else "Bolzmann"
        for i, node in enumerate(self.nodes):
            self.add_module(node.name if hasattr(node, "name") else f"node{i}", node)
        self.device = device
        self.to(device)

    def forward(self, inputs, labels):
        outputs = [node(inputs, labels) for node in self.nodes]
        return torch.mean(torch.stack(outputs), dim=0)

    def update(self, inputs, labels, states):
        outputs = [
            torch.zeros(inputs.shape, dtype=inputs.dtype, device=inputs.device)
        ] * len(self.nodes)
        losses = []
        for _ in range(self.bounces + 1):
            temp = []
            for nnum in range(len(self.nodes)):
                echo = torch.mean(
                    torch.stack(outputs[:nnum] + outputs[nnum + 1 :]), dim=0
                )
                temp[nnum], loss = self.nodes[nnum].update(
                    inputs + echo, labels, states
                )
                losses.append(loss.item())
            outputs = [el for el in temp.detach()]

    def __str__(self) -> str:
        output = "Bolzmann(\n"
        for node in self.nodes:
            output += f" ({node.name}): " + node.__str__() + "\n"
        output += ")\n"
        return output
