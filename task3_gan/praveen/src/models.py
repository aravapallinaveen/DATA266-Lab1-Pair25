"""Praveen SID 8511 CycleGAN models."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def init_weights(module):
    name = module.__class__.__name__
    if "Conv" in name and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in name and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=48, n_blocks=6):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        layers = [
            nn.ReflectionPad2d(3), nn.Conv2d(in_channels, c1, 7),
            nn.InstanceNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.InstanceNorm2d(c3), nn.ReLU(inplace=True),
        ]
        layers += [ResidualBlock(c3) for _ in range(n_blocks)]
        layers += [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c2, 3, padding=1), nn.InstanceNorm2d(c2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, 3, padding=1), nn.InstanceNorm2d(c1), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3), nn.Conv2d(c1, out_channels, 7), nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=48, use_spectral_norm=True):
        super().__init__()
        def conv(cin, cout, stride, norm=True):
            layer = nn.Conv2d(cin, cout, 4, stride=stride, padding=1)
            if use_spectral_norm:
                layer = spectral_norm(layer)
            result = [layer]
            if norm:
                result.append(nn.InstanceNorm2d(cout))
            result.append(nn.LeakyReLU(0.2, inplace=True))
            return result
        layers = []
        layers += conv(in_channels, base, 2, norm=False)
        layers += conv(base, base * 2, 2)
        layers += conv(base * 2, base * 4, 2)
        layers += conv(base * 4, base * 8, 1)
        final = nn.Conv2d(base * 8, 1, 4, stride=1, padding=1)
        self.net = nn.Sequential(*layers, spectral_norm(final) if use_spectral_norm else final)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    """Two PatchGAN discriminators operating at full and half resolution."""
    def __init__(self, in_channels=3, base=48, use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(
                in_channels=in_channels,
                base=base,
                use_spectral_norm=use_spectral_norm,
            ),
            Discriminator(
                in_channels=in_channels,
                base=base,
                use_spectral_norm=use_spectral_norm,
            ),
        ])

    def forward(self, x):
        outputs = []
        current = x

        for index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(current))
            if index + 1 < len(self.discriminators):
                current = F.avg_pool2d(
                    current,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=False,
                )

        return outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

