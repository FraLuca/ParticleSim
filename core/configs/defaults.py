import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "deeplabv2_resnet101"
_C.MODEL.NUM_CLASSES = 19
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False
_C.MODEL.HYPER = False
_C.MODEL.CURVATURE = 1.
_C.MODEL.REDUCED_CHANNELS = 256
_C.MODEL.WEIGHTED_NORM = False

_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.NAME = 'test_run'
_C.WANDB.GROUP = 'deeplabv2_r101_pretrain'
_C.WANDB.PROJECT = 'active_domain_adapt'
_C.WANDB.ENTITY = 'pinlab-sapienza'

_C.INPUT = CN()
_C.INPUT.DATA_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.DATA_STD = [0.229, 0.224, 0.225]

_C.DATASETS = CN()
_C.DATASETS.TRAIN = ""
_C.DATASETS.VAL = ""
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.GPUS = [0,1,2,3]
_C.SOLVER.NUM_EPOCH = 200
_C.SOLVER.CHECKPOINT_PERIOD = 50

# optimizer and learning rate
_C.SOLVER.LR_SCHEDULER = 'poly'
_C.SOLVER.BASE_LR = 2.5e-4
_C.SOLVER.LR_POWER = 0.9
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005

# 4 images per batch, two for source and two for target
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.BATCH_SIZE_VAL = 1

# hyper-parameters
_C.SOLVER.CONSISTENT_LOSS = 0.
_C.SOLVER.NEGATIVE_LOSS = 1.0
_C.SOLVER.NEGATIVE_THRESHOLD = 0.05

# local consistent loss
_C.SOLVER.LCR_TYPE = "l1"


_C.ACTIVE = CN()
# active strategy
_C.ACTIVE.NAME = 'AL-RIPU'
_C.ACTIVE.UNCERTAINTY = 'entropy'
_C.ACTIVE.PURITY = 'hyper'
_C.ACTIVE.SETTING = 'RA'
_C.ACTIVE.SELECT_ITER = [0, 15001, 30001, 40001, 50001] # for 5 selection rounds
# total selection ratio per image
_C.ACTIVE.RATIO = 0.022
# total selected pixel per image
_C.ACTIVE.PIXELS = 40
# hyper-parameters
_C.ACTIVE.RADIUS_K = 1
_C.ACTIVE.NORMALIZE = True
_C.ACTIVE.MASK_RADIUS_K = 5
_C.ACTIVE.K = 100
_C.ACTIVE.QUANT = 'uniform'
# visualization of active selection
_C.ACTIVE.VIZ_MASK = True


# ---------------------------------------------------------------------------- #
# Specific val options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 128


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.resume = ""
_C.checkpoint = ""
_C.SEED = -1
_C.NUM_WORKERS = 8
_C.DEBUG = 0
_C.PROTOCOL = "source"
