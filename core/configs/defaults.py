import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = "mlp"
_C.MODEL.ENCODER.INPUT_SIZE = 47
_C.MODEL.ENCODER.OUTPUT_SIZE = 512
_C.MODEL.ENCODER.DROPOUT = 0.2
_C.MODEL.ENCODER.NHEAD = 8
_C.MODEL.ENCODER.HIDDEN_SIZE = 512
_C.MODEL.ENCODER.NUM_LAYERS = 6
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.CLASSIFIER = CN()
_C.MODEL.HEAD.CLASSIFIER.HIDDEN_SIZE = 128
_C.MODEL.HEAD.CLASSIFIER.OUTPUT_SIZE = 1        # the single p probability when using BCELogitsLoss
_C.MODEL.HEAD.CLASSIFIER.DROPOUT = 0.2
_C.MODEL.HEAD.REGRESSOR = CN()
_C.MODEL.HEAD.REGRESSOR.HIDDEN_SIZE = 128
_C.MODEL.HEAD.REGRESSOR.OUTPUT_SIZE = 2
_C.MODEL.HEAD.REGRESSOR.DROPOUT = 0.2


_C.INPUT = CN()
_C.INPUT.DATA_MEAN = [0, 0, 0, 0, 0, 0, 609.8382, 652.1469, 674.2386, 603.385, 485.9594, 556.3623, 431.7391, 411.7426, -0.5149, 349.3276, 289.8358, 265.9223, 228.8163, 237.0586, 192.8511, 185.3474, 152.9026, 156.7927, 136.2225, 120.4612, 107.4664, 115.0627, 91.6162, 84.3827, 70.2917, 77.1347, 61.448, 58.896, 50.5762, 50.9453, 44.4321, 42.6175, 1.8524, 2.0049, 1.8132, 2.0478, 2.0034, 1.8143, 1.8414, 1.9082, 1.9826, 34.8490, 286.8785]
_C.INPUT.DATA_STD = [1, 1, 1, 1, 1 ,1, 819.281, 843.3945, 922.5032, 903.6316, 643.5854, 870.7497, 662.6011, 657.2284, 11.0661, 755.6806, 679.1472, 620.8115, 507.723, 533.1241, 495.2585, 472.719, 493.5004, 437.9316, 449.7092, 385.9413, 410.6029, 408.5552, 353.0984, 343.6835, 282.9658, 320.7973, 263.2151, 218.7552, 215.5216, 209.6305, 206.2789, 199.1454, 23.1784, 23.1114, 23.0327, 23.5193, 24.0979, 23.329, 24.4089, 23.7204, 23.6168, 28.5176, 279.123]

_C.DATASETS = CN()
_C.DATASETS.TRAIN = "data/dataset_train.csv"
_C.DATASETS.VAL = "data/dataset_validation.csv"
_C.DATASETS.TEST = "data/dataset_test.csv"

_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 300
_C.SOLVER.BATCH_SIZE = 2048
_C.SOLVER.BASE_LR = 1.5e-3
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.GPUS = [0]
_C.SOLVER.OPTIMIZER = 'AdamW'
_C.SOLVER.LR_SCHEDULER = 'MultiStepLR'
_C.SOLVER.MILESTONES = [100,200]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.MIN_LR = 1e-6

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LR_POWER = 0.9
_C.SOLVER.MOMENTUM = 0.9


_C.WANDB = CN()
_C.WANDB.ENABLE = True
_C.WANDB.NAME = 'debug'
_C.WANDB.GROUP = 'source'
_C.WANDB.PROJECT = 'uda_particle_simul'
_C.WANDB.ENTITY = 'particlesim'


# ---------------------------------------------------------------------------- #
# Specific val options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 2048

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "results/source/debug"
_C.PROTOCOL = "source"
_C.SEED = -1
_C.NUM_WORKERS = 8

_C.resume = ""
_C.checkpoint = ""
_C.DEBUG = 0

