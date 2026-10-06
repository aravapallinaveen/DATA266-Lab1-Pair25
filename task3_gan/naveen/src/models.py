"""Conventional ResNet-9 generator and 70x70 PatchGAN, trained from scratch."""
from torch import nn

def norm(channels):
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
                                 norm(channels), nn.ReLU(True), nn.ReflectionPad2d(1),
                                 nn.Conv2d(channels, channels, 3), norm(channels))
    def forward(self, x):
        return x + self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), norm(64), nn.ReLU(True)]
        for c in (64, 128):
            layers += [nn.Conv2d(c, c*2, 3, stride=2, padding=1), norm(c*2), nn.ReLU(True)]
        layers += [ResidualBlock(256) for _ in range(9)]
        for c in (256, 128):
            layers += [nn.ConvTranspose2d(c, c//2, 3, stride=2, padding=1,
                                         output_padding=1), norm(c//2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(.2, True)]
        for ci, co, stride in ((64, 128, 2), (128, 256, 2), (256, 512, 1)):
            layers += [nn.Conv2d(ci, co, 4, stride=stride, padding=1), norm(co),
                       nn.LeakyReLU(.2, True)]
        layers += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0., .02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def build_models(device):
    models = dict(G_A2B=Generator(), G_B2A=Generator(), D_A=Discriminator(), D_B=Discriminator())
    for model in models.values():
        model.apply(initialize)
        model.to(device)
    return models
