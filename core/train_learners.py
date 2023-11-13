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
from core.model.encoder import build_encoder
from core.model.heads import build_classifier, build_regressor



# class BaseLearner(pl.LightningModule):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg

#         # create network
#         self.encoder = build_encoder(cfg)
#         self.classifier = build_classifier(cfg)
#         self.regressor = build_regressor(cfg)

#         # create criterion
#         self.class_loss = nn.CrossEntropyLoss()
#         self.regress_loss = nn.MSELoss()

#     def forward(self, input_data):
#         embedding = self.encoder(input_data)
#         part_pred = self.classifier(embedding)
#         energy_pred = self.regressor(embedding)
#         return energy_pred, part_pred

#     def inference(self, input_data):
#         embedding = self.encoder(input_data)
#         part_pred = self.classifier(embedding)
#         energy_pred = self.regressor(embedding)
#         part_pred_logits = F.softmax(part_pred, dim=-1)
#         return energy_pred, part_pred_logits

#     def validation_step(self, batch, batch_idx):

#         signal, gt_energy, gt_particle_type = batch[0], batch[1], batch[2]   # shapes [B, 47], [B, 1], [B, 1]
#         energy_pred, part_pred_logits = self.inference(signal)
        
#         part_pred_class = part_pred_logits.argmax(dim=-1)

#         val_particle_acc = (part_pred_class == gt_particle_type).float().mean()
#         self.log('val_particle_acc', val_particle_acc.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

#         error_energy = torch.abs(energy_pred - gt_energy).mean()
#         self.log('val_energy_error', error_energy.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)


#     def configure_optimizers(self):
#         if self.cfg.SOLVER.OPTIMIZER == 'SGD':
#             optimizer = SGD(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, 
#                             momentum=self.cfg.SOLVER.MOMENTUM, 
#                             weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
#         elif self.cfg.SOLVER.OPTIMIZER == 'AdamW':
#             optimizer = AdamW(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, 
#                             weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

#         if self.cfg.SOLVER.LR_SCHEDULER == "MultiStepLR":
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(
#                 optimizer,
#                 milestones=self.cfg.SOLVER.MILESTONES,
#                 gamma=self.cfg.SOLVER.GAMMA,
#             )
#         elif self.cfg.SOLVER.LR_SCHEDULER == "CosineAnnealingLR":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=self.cfg.SOLVER.EPOCHS, 
#                 eta_min=self.cfg.SOLVER.MIN_LR
#             )
#         else:
#             scheduler = None

#         return [optimizer], [scheduler]
    

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




# TODO: adapt this to the phase 2
# class SourceTargetLearner(SourceFreeLearner):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#         self.local_consistent_loss = LocalConsistentLoss(cfg.MODEL.NUM_CLASSES, cfg.SOLVER.LCR_TYPE)

#     def training_step(self, batch, batch_idx):
#         optimizers = self.optimizers()
#         for opt in optimizers:
#             opt.zero_grad()

#         # source data
#         src_input, src_label = batch[0]['img'], batch[0]['label']
#         src_out = self.forward(src_input)
#         if self.hyper:
#             src_out = src_out[0]
#         else:
#             src_out = src_out[0]

#         # target data
#         # tgt_mask is active label, 255 means unlabeled data
#         tgt_input, tgt_mask = batch[1]['img'], batch[1]['mask']
#         tgt_out = self.forward(tgt_input)
#         if self.hyper:
#             tgt_out = tgt_out[0]
#         else:
#             tgt_out = tgt_out[0]

#         predict = torch.softmax(tgt_out, dim=1)
#         loss = torch.Tensor([0]).cuda()

#         # source supervision loss
#         loss_sup = self.criterion(src_out, src_label)
#         loss += loss_sup
#         self.log('loss_sup', loss_sup.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

#         # target active supervision loss
#         if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
#             loss_sup_tgt = self.criterion(tgt_out, tgt_mask)
#             loss += loss_sup_tgt
#             self.log('loss_sup_tgt', loss_sup_tgt.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

#         # source consistency regularization loss
#         if self.cfg.SOLVER.CONSISTENT_LOSS > 0:
#             consistency_loss = self.local_consistent_loss(src_out, src_label) * self.cfg.SOLVER.CONSISTENT_LOSS
#             loss += consistency_loss
#             self.log('consistency_loss', consistency_loss.item(), on_step=False,
#                      on_epoch=True, sync_dist=True, prog_bar=True)

#         # target negative pseudo loss
#         if self.cfg.SOLVER.NEGATIVE_LOSS > 0:
#             negative_loss = self.negative_criterion(predict) * self.cfg.SOLVER.NEGATIVE_LOSS
#             loss += negative_loss
#             self.log('negative_loss', negative_loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

#         self.log('loss', loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
#         self.log_metrics(batch_idx)

#         # manual backward pass
#         self.manual_backward(loss)
#         for opt in optimizers:
#             opt.step()
#         for sched in self.lr_schedulers():
#             sched.step()

#     def train_dataloader(self):
#         source_set = build_dataset(self.cfg, mode='train', is_source=True)
#         target_set = build_dataset(self.cfg, mode='train', is_source=False)
#         self.data_len = len(source_set)
#         self.target_len = len(target_set)
#         self.print('source data length: ', self.data_len)
#         self.print('target data length: ', self.target_len)
#         source_loader = DataLoader(
#             dataset=source_set,
#             batch_size=self.cfg.SOLVER.BATCH_SIZE,
#             shuffle=True,
#             num_workers=NUM_WORKERS,
#             pin_memory=True,
#             drop_last=True,
#             persistent_workers=True,)
#         target_loader = DataLoader(
#             dataset=target_set,
#             batch_size=self.cfg.SOLVER.BATCH_SIZE,
#             shuffle=True,
#             num_workers=NUM_WORKERS,
#             pin_memory=True,
#             drop_last=True,
#             persistent_workers=True,)
#         return [source_loader, target_loader]


# TODO: adapt this for the considered task. 
# Do we need one for phase 1 and one for phase 2?
# class Test(BaseLearner):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#         # evaluation metrics
#         self.intersections = np.array([])
#         self.unions = np.array([])
#         self.targets = np.array([])

#     def test_step(self, batch, batch_idx):
#         x, y = batch['img'], batch['label']

#         if 'name' in batch.keys():
#             name = batch['name']
#             name = name[0]
#             name = name.rsplit('/', 1)[-1].rsplit('_', 1)[0]
#         else:
#             name = str(batch_idx)

#         embed_file_name = None
#         if self.cfg.TEST.SAVE_EMBED:
#             self.save_embeddings(y, name, 'label')
#             embed_file_name = os.path.join(self.cfg.OUTPUT_DIR, 'embed', name + '.pt')

#         wrong_file_name = None
#         if self.cfg.TEST.VIZ_WRONG and (batch_idx in VIZ_LIST):
#             wrong_file_name = os.path.join(self.cfg.OUTPUT_DIR, 'viz', 'wrong', name + '.png')

#         output = self.inference(x, y, flip=True, save_embed_path=embed_file_name,
#                               save_wrong_path=wrong_file_name, cfg=self.cfg)
#         pred = output.max(1)[1]

#         if self.cfg.TEST.SAVE_EMBED:
#             self.save_embeddings(pred, name, 'pred')
#             self.save_embeddings(output, name, 'output')

#         intersection, union, target = self.intersectionAndUnionGPU(
#             pred, y, self.cfg.MODEL.NUM_CLASSES, self.cfg.INPUT.IGNORE_LABEL)

#         intersection = np.expand_dims(intersection, axis=0)
#         union = np.expand_dims(union, axis=0)
#         target = np.expand_dims(target, axis=0)

#         if self.intersections.size == 0:
#             self.intersections = intersection
#             self.unions = union
#             self.targets = target
#         else:
#             self.intersections = np.concatenate((self.intersections, intersection), axis=0)
#             self.unions = np.concatenate((self.unions, union), axis=0)
#             self.targets = np.concatenate((self.targets, target), axis=0)

#     def on_test_epoch_end(self):
#         # gather all the metrics across all the processes
#         intersections = self.all_gather(self.intersections)
#         unions = self.all_gather(self.unions)
#         targets = self.all_gather(self.targets)

#         intersections = intersections.flatten(0, 1)
#         unions = unions.flatten(0, 1)
#         targets = targets.flatten(0, 1)

#         # calculate the final mean iou and accuracy
#         intersections = self.intersections.sum(axis=0)
#         unions = self.unions.sum(axis=0)
#         targets = self.targets.sum(axis=0)

#         iou_class = intersections / (unions + 1e-10)
#         accuracy_class = intersections / (targets + 1e-10)

#         mIoU = round(iou_class.mean() * 100, 2)
#         mAcc = round(accuracy_class.mean() * 100, 2)
#         aAcc = round(intersections.sum() / (targets.sum() + 1e-10) * 100, 2)

#         # print IoU per class
#         print('\n\n')
#         print('{:<20}  {:<20}  {:<20}'.format('Class', 'IoU (%)', 'Accuracy (%)'))
#         for i in range(cfg.MODEL.NUM_CLASSES):
#             print('{:<20}  {:<20.2f}  {:<20.2f}'.format(self.class_names[i], iou_class[i] * 100, accuracy_class[i] * 100))

#         # print mIoUs in LateX format
#         print()
#         print('mIoU in LateX format:')
#         delimiter = ' & '
#         latex_iou_class = delimiter.join(map(lambda x: '{:.1f}'.format(x*100), iou_class))
#         print(latex_iou_class)

#         # print metrics table style
#         print()
#         print('mIoU:\t {:.2f}'.format(mIoU))
#         print('mAcc:\t {:.2f}'.format(mAcc))
#         print('aAcc:\t {:.2f}\n'.format(aAcc))

#         # log metrics
#         self.log('mIoU', mIoU, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
#         self.log('mAcc', mAcc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
#         self.log('aAcc', aAcc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)

#     def test_dataloader(self):
#         test_set = build_dataset(self.cfg, mode='test', is_source=False)
#         self.class_names = test_set.trainid2name
#         test_loader = DataLoader(
#             dataset=test_set,
#             batch_size=1,
#             shuffle=False,
#             num_workers=8,
#             pin_memory=True,
#             drop_last=False,)
#         return test_loader

#     def save_embeddings(self, output, name, type='embed'):
#         dir_path = os.path.join(self.cfg.OUTPUT_DIR, type)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#         file_name = os.path.join(dir_path, name + '.pt')
#         torch.save(output.cpu(), file_name)
