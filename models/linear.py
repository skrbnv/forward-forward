from libs.ffblocks import FFLinearBlock
from libs.ffmodel import FFModel, FFModelAfterInit


class Model(FFModel, metaclass=FFModelAfterInit):
    def __init__(
        self,
        device="cpu",
        inner_sizes: list = [28 * 28, 256, 256],
        num_classes: int = 10,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(device=device, *args, **kwargs)
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
        for i, layer in enumerate(self.layers):
            self.add_module(
                layer.name if hasattr(layer, "name") else f"layer{i}", layer
            )
