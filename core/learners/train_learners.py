import os
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

from core.configs import cfg
from core.datasets.build import build_dataset
from core.model.encoder import build_encoder, load_encoder
from core.model.heads import build_classifier, build_regressor
from core.model.gan import Generator, Discriminator




class SourceLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # create network
        self.encoder = build_encoder(cfg)
        self.classifier = build_classifier(cfg)
        self.regressor = build_regressor(cfg)

        # create criterion
        # self.class_loss = nn.CrossEntropyLoss()
        self.class_loss = nn.BCEWithLogitsLoss()
        self.regress_loss = nn.MSELoss()

    def forward(self, input_data):
        embedding = self.encoder(input_data)
        part_pred = self.classifier(embedding)
        energy_pred = self.regressor(embedding)
        return energy_pred, part_pred

    def training_step(self, batch, batch_idx):

        signal, gt_energy, gt_particle_type = batch[0].float(), batch[1].float(), batch[2].long()   # shapes [B, 53], [B, 1], [B, 1]
        energy_pred, part_pred_logits = self.forward(signal)

        loss_cls = self.class_loss(pred_particle_type, gt_particle_type.squeeze(-1))
        loss_reg = self.regress_loss(pred_energy, gt_energy)
        loss = 1. * loss_cls + 1. * loss_reg

        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_loss_cls', loss_cls.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log('train_loss_reg', loss_reg.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log('train_accuracy', train_accuracy.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def inference(self, input_data):
        embedding = self.encoder(input_data)
        part_pred = self.classifier(embedding)
        energy_pred = self.regressor(embedding)
        part_pred_logits = F.softmax(part_pred, dim=-1)
        return energy_pred, part_pred_logits

    def validation_step(self, batch, batch_idx):

        signal, gt_energy, gt_particle_type = batch[0].float(), batch[1].float(), batch[2].float()   # shapes [B, 47], [B, 1], [B, 1]
        energy_pred, part_pred_logits = self.inference(signal)
    
        # part_pred_class = part_pred_logits.argmax(dim=-1)

        # val_particle_acc = (part_pred_class == gt_particle_type).float().mean()
        val_particle_acc = ((part_pred_logits > 0.5) == gt_particle_type.squeeze(-1)).float().mean()
        self.log('val_particle_acc', val_particle_acc.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        error_energy = torch.abs(energy_pred - gt_energy).mean()
        self.log('val_energy_error', error_energy.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

    def train_dataloader(self):
        train_set = build_dataset(self.cfg, mode='train')
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader


    def val_dataloader(self):
        val_set = build_dataset(self.cfg, mode='val')
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg.VAL.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,)
        return val_loader


    def configure_optimizers(self):
        if self.cfg.SOLVER.OPTIMIZER == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, 
                            momentum=self.cfg.SOLVER.MOMENTUM, 
                            weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.OPTIMIZER == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, 
                            weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

        if self.cfg.SOLVER.LR_SCHEDULER == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.SOLVER.MILESTONES,
                gamma=self.cfg.SOLVER.GAMMA,
            )
        elif self.cfg.SOLVER.LR_SCHEDULER == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.SOLVER.EPOCHS, 
                eta_min=self.cfg.SOLVER.MIN_LR
            )
        else:
            scheduler = None

        return [optimizer], [scheduler]






class TargetLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)

        self.encoder = load_encoder(cfg)


    def forward(self, z):
        return self.encoder(self.generator(z))

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        signal, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(signal.shape[0], self.cfg.MODEL.GENERATOR.INPUT_SIZE)
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
    
    def train_dataloader(self):
        train_set = build_dataset(self.cfg, mode='train', domain='target')
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader
