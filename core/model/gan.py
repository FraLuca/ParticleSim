import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.encoder import build_encoder, load_encoder
import pytorch_lightning as L



class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.cfg.MODEL.GENERATOR.INPUT_SIZE, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.cfg.MODEL.GENERATOR.OUTPUT_SIZE),
            nn.Tanh(),
        )

    def forward(self, z):
        signal = self.model(z)
        signal = signal.view(signal.size(0), *self.signal_shape)
        return signal



class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = nn.Sequential(
            nn.Linear(self.cfg.MODEL.ENCODER.OUTPUT_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal):
        signal_flat = signal.view(signal.size(0), -1)
        validity = self.model(signal_flat)
        return validity










