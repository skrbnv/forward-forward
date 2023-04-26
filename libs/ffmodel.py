from torch import no_grad
import torch.nn as nn
from libs.ffblocks import is_ff


class FFModelAfterInit(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.after_init()
        return instance


class FFModel(nn.Module):
    def __init__(
        self,
        device="cpu",
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.scale = scale
        self.layers = []

    def after_init(self):
        for i, layer in enumerate(self.layers):
            self.add_module(
                layer.name if hasattr(layer, "name") else f"layer{i}", layer
            )

    def update(self, inputs, labels, states):
        x = inputs / self.scale
        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                x, loss = layer.update(x, labels, states)
                losses += [loss.item()]
            else:
                with no_grad():
                    x = layer(x)
        return x, losses

    def update_layer(self, inputs, labels, states, layer_num):
        x = inputs / self.scale
        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                if i == layer_num:
                    x, loss = layer.update(x, labels, states)
                    losses += [loss.item()]
                else:
                    with no_grad():
                        x = layer(x, labels)
            else:
                with no_grad():
                    x = layer(x)
        return x, losses

    def compute(self, inputs, labels):
        x = inputs / self.scale
        with no_grad():
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
        def calc(x):
            if len(x.shape) == 4:
                output = (x**2).sum(dim=(2, 3)).mean(dim=1)
            elif len(x.shape) == 2:
                output = (x**2).sum(dim=1)
            else:
                raise Exception("Unknown shape")
            return output

        x = inputs / self.scale
        goodness = None
        with no_grad():
            for layer in self.layers:
                if is_ff(layer):
                    x = layer(x, labels)
                    goodness = calc(x) if goodness is None else goodness + calc(x)
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
