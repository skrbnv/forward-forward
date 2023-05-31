from torch import no_grad
import torch.nn as nn
from libs.ffblocks import is_ff, metric


class FFModelAfterInit(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.after_init()
        return instance


class FFModel(nn.Module):
    def __init__(
        self,
        device="cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.layers = []

    def after_init(self):
        for i, layer in enumerate(self.layers):
            self.add_module(
                layer.name if hasattr(layer, "name") else f"layer{i}", layer
            )

    def update(self, inputs, labels, states):
        x = inputs
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
        x = inputs
        losses = []
        for i, layer in enumerate(self.layers):
            if is_ff(layer):
                if i == layer_num:
                    x, loss = layer.update(x, labels, states)
                    losses += [loss.item()]
                    break
                else:
                    with no_grad():
                        x = layer(x, labels)
            else:
                with no_grad():
                    x = layer(x)
        return x, losses

    def reschedule(self):
        lrs = []
        for i in range(len(self.layers)):
            if is_ff(self.layers[i]):
                self.layers[i].scheduler.step()
                lrs += [f"{self.layers[i].optimizer.param_groups[0]['lr']:.6f}"]
        print(f"  New LR[0] are {', '.join(lrs)}")

    def compute(self, inputs, labels):
        x = inputs
        with no_grad():
            for layer in self.layers:
                if is_ff(layer):
                    x = layer(x, labels)
                else:
                    x = layer(x)
        return x

    def goodness_last_layer(self, inputs, labels):
        x = self.compute(inputs, labels)
        return metric(x)

    def goodness(self, inputs, labels):
        x = inputs
        goodness = None
        with no_grad():
            for layer in self.layers:
                if is_ff(layer):
                    x = layer(x, labels)
                    goodness = metric(x) if goodness is None else goodness + metric(x)
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
