import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.encoder import build_encoder



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
            *block(self.cfg.GENERATOR.INPUT_SIZE, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.cfg.GENERATOR.OUTPUT_SIZE),
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
            nn.Linear(self.cfg.DISCRIMINATOR.INPUT_SIZE, 512),
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





class GAN(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)

        self.encoder = build_encoder(cfg)
        # load pretrained source encoder


    def forward(self, z):
        return self.encoder(self.generator(z))

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        signal, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(signal.shape[0], self.cfg.GENERATOR.INPUT_SIZE)
        z = z.type_as(signal)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(signal.size(0), 1)
        valid = valid.type_as(signal)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(signal.size(0), 1)
        valid = valid.type_as(signal)

        real_loss = self.adversarial_loss(self.discriminator(signal), valid)

        # how well can it label as fake?
        fake = torch.zeros(signal.size(0), 1)
        fake = fake.type_as(signal)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)


    def configure_optimizers(self):
        lr = self.cfg.SOLVER.BASE_LR
        b1 = self.cfg.SOLVER.B1
        b2 = self.cfg.SOLVER.B2

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []






