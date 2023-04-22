from libs.ffblocks import is_ff
import torch


class FFDispatcher:
    def __init__(self, object) -> None:
        self.object = object

    def follower(self, index):
        assert index < self.object.layer_count(), "Index too large"
        for i in range(index + 1, len(self.object.layers)):
            if is_ff(self.object.layers[i]):
                return i
        else:  # no follower
            return None

    def follower_response(self, index, inputs, labels):
        if self.follower(index) is None:
            return None
        for i in range(index + 1, self.follower(index) + 1):
            if is_ff(self.object.layers[i]):
                outputs = self.object.layers[i].compute(inputs, labels)
                return self.goodness(outputs)
            else:
                inputs = self.object.layers[i](inputs)
        else:
            raise Exception("Unreachable point reached")

    def goodness(self, inputs):
        if len(inputs.shape) == 4:
            return torch.sum(inputs**2, dim=(-2, -1)).mean(-1) / (
                inputs.size(-1) * inputs.size(-2)
            )
        elif len(inputs.shape) == 2:
            return torch.sum(inputs**2, dim=-1) / inputs.size(-1)
        else:
            raise Exception(f"Cannot compute goodness for shape {inputs.shape}")
