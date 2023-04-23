import torch
import torch.nn as nn
from libs.ffblocks import FFLinearBlock, FFConvBlock, is_ff
from libs.dispatcher import FFDispatcher


class FFLinearModel:
    def __init__(
        self,
        device="cpu",
        inner_sizes: list = [28 * 28, 100, 100, 10],
        num_classes: int = 10,
    ) -> None:
        self.device = device
        self.layers = []
        for i in range(len(inner_sizes) - 1):
            self.layers.append(
                FFLinearBlock(
                    ins=inner_sizes[i],
                    outs=inner_sizes[i + 1],
                    device=device,
                    num_classes=num_classes,
                    name=f"L{i}",
                )
            )

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
                with torch.no_grad():
                    x = layer(x, labels)
        return x, losses

    def compute(self, inputs, labels):
        x = inputs
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x, labels)
        return x

    def goodness(self, inputs, labels):
        x = self.compute(inputs, labels)
        return (x**2).sum(dim=1)

    def layer_count(self):
        return len(self.layers)


class FFConvModel(nn.Module):
    def __init__(
        self,
        device="cpu",
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.device = device
        self.layers = []
        self.layers += [
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
            FFLinearBlock(
                ins=3136,
                outs=256,
                device=device,
                num_classes=num_classes,
                name="dense1",
            ),
            FFLinearBlock(
                ins=256,
                outs=10,
                device=device,
                num_classes=num_classes,
                name="dense2",
            ),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(
                layer.name if hasattr(layer, "name") else f"layer{i}", layer
            )
        self.dispatcher = FFDispatcher(self)

    def update(self, inputs, labels, statuses):
        x = inputs
        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                x, loss = layer.update(x, labels, statuses)
                losses += [loss.item()]
            else:
                with torch.no_grad():
                    x = layer(x)
        return x, losses

    def update_layer(self, inputs, labels, statuses, layer_num):
        x = inputs
        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                if i == layer_num:
                    x, loss = layer.update(x, labels, statuses)
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
                (x**2).sum(dim=(2, 3)).mean(-1)
                if len(x.shape) == 4
                else (x**2).sum(dim=-1)
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
