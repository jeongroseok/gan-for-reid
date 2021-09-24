from typing import Tuple
from torch import nn
import numpy as np


class BasicClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: int,
        *args: any,
        **kwargs: any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


class BasicDiscriminator(nn.Module):
    def __init__(
        self,
        img_dim: Tuple[int, int, int],
        num_classes: int,
        first_hidden_dim: int = 64,
        normalization: bool = True,
    ):
        super().__init__()

        def block(in_channels, out_channels, normalization: bool = normalization):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv = nn.Sequential(
            *block(img_dim[0], first_hidden_dim, False),
            *block(first_hidden_dim, first_hidden_dim * 2),
            *block(first_hidden_dim * 2, first_hidden_dim * 4),
            *block(first_hidden_dim * 4, first_hidden_dim * 8),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(first_hidden_dim * 8, 1, kernel_size=1), nn.Sigmoid(),
        )  # PatchGAN
        self.classifier = BasicClassifier(num_classes, first_hidden_dim * 8, 256)

    def forward(self, x):
        x = self.conv(x)
        d = self.fc(x)  # 판별값
        y_hat = self.classifier(x)  # 분류값
        return d, y_hat
