import torch.nn as nn
from libs.ffblocks import FFLinearBlock, FFConvBlock
from libs.ffmodel import FFModel, FFModelAfterInit


class Model(FFModel, metaclass=FFModelAfterInit):
    def __init__(self, device="cpu", num_classes: int = 10, *args, **kwargs) -> None:
        super().__init__(device=device, *args, **kwargs)
        self.layers += [
            FFConvBlock(
                channels_in=1,
                channels_out=64,
                shape_in=(28, 28),
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
                norm=nn.LayerNorm(3136),
                name="dense1",
            ),
            FFLinearBlock(
                ins=256,
                outs=10,
                device=device,
                num_classes=num_classes,
                norm=nn.LayerNorm(256),
                name="dense1",
            ),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(
                layer.name if hasattr(layer, "name") else f"layer{i}", layer
            )
