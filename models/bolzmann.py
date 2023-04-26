import torch
import torch.nn as nn
from libs.ffblocks import FFLinearBlock, FFConvBlock, FFBolzmannChain, is_ff
from libs.ffmodel import FFModel, FFModelAfterInit


class Model(FFModel, metaclass=FFModelAfterInit):
    def __init__(self, device="cpu", num_classes: int = 10, *args, **kwargs) -> None:
        super().__init__(device=device, *args, **kwargs)
        self.layers = [
            FFConvBlock(
                channels_in=1,
                channels_out=64,
                kernel_size=3,
                stride=2,
                padding=1,
                device=device,
                norm=nn.Identity(),
                name="conv1",
                num_classes=num_classes,
            ),
            nn.AvgPool2d(2),
            nn.Flatten(),
            FFBolzmannChain(
                bounces=3,
                device=self.device,
                nodes=[
                    FFLinearBlock(
                        ins=3136,
                        outs=3136,
                        device=device,
                        num_classes=num_classes,
                        name="bzmnn",
                    )
                ]
                * 3,
            ),
        ]

    def update(self, inputs, labels, states):
        x = inputs / self.scale
        x = self.prepare(x)

        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                x, loss = layer.update(x, labels, states)
                losses += [loss.item()]
            else:
                with torch.no_grad():
                    x = layer(x)
        return x, losses

    def update_layer(self, inputs, labels, states, layer_num):
        x = inputs
        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                if i == layer_num:
                    x, loss = layer.update(x, labels, states)
                    losses += [loss.item()]
                    """
                    with torch.no_grad():
                        response = self.dispatcher.follower_response(
                            layer_num, x.clone().detach(), labels
                        )
                        if response is not None:
                            assert (
                                response.size(0) == x.size(0)
                                and len(response.shape) == 1
                            ), "Follower response is incorrect"
                            for i in range(len(x.shape) - 1):
                                response = response.unsqueeze(-1)
                            x = x + response
                    """
                else:
                    with torch.no_grad():
                        x = layer(x, labels)
            else:
                with torch.no_grad():
                    x = layer(x)
        return x, losses

    def compute(self, inputs, labels):
        x = inputs
        with torch.no_grad():
            for layer in self.layers:
                if is_ff(layer):
                    x = layer(x, labels)
                else:
                    x = layer(x)
        return x

    def goodness_last_layer(self, inputs, labels):
        x = self.compute(inputs, labels)
        return (x**2).sum(dim=1)

    def goodness(self, inputs, labels):
        def eval(x):
            return (
                (x**2).sum(dim=(2, 3)).mean(dim=1)
                if len(x.shape) == 4
                else (x**2).sum(dim=1)
            )

        x = inputs
        goodness = None
        with torch.no_grad():
            for layer in self.layers:
                if is_ff(layer):
                    x = layer(x, labels)
                    goodness = eval(x) if goodness is None else goodness + eval(x)
                else:
                    x = layer(x)
        return goodness

    def layer_count(self):
        return len(self.layers)

    def __str__(self) -> str:
        output = ""
        for layer in self.layers:
            output += layer.__str__() + "\n"
        return output
