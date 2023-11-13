from pathlib import Path
import warnings
import numpy as np
from core.utils.misc import parse_args
from core.configs import cfg
from core.train_learners import SourceLearner #, SourceTargetLearner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
import lightning as L

warnings.filterwarnings('ignore')


# protocol_types = {
#     'source': SourceLearner #,
#     # 'source_target': SourceTargetLearner
# }


def main(cfg):

    # init wandb logger
    wandb_logger = None
    if cfg.WANDB.ENABLE:
        wandb_logger = WandbLogger(project=cfg.WANDB.PROJECT, name=cfg.WANDB.NAME,
                                   entity=cfg.WANDB.ENTITY, group=cfg.WANDB.GROUP,
                                   config=cfg, save_dir='.')

    # init learner
    if cfg.PROTOCOL == 'source':
        learner = SourceLearner(cfg)
    else:
        raise NotImplementedError(f'Protocol {cfg.PROTOCOL} is not implemented.')

    checkcall_1 = ModelCheckpoint(
        save_top_k=1,
        monitor="val_energy_error",  # TODO: change with metric for energy regression or particle classification
        mode="min",
        dirpath=cfg.OUTPUT_DIR,
        filename="model_{global_step}_{val_energy_error:.2f}",  # TODO: change consistently
    )
    
    # init trainer
    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.EPOCHS,
        accelerator='gpu',
        devices=cfg.SOLVER.GPUS,
        strategy="ddp",  # "ddp_find_unused_parameters_true",
        logger=wandb_logger,
        callbacks=[checkcall_1],
        num_sanity_val_steps=2,
        sync_batchnorm=True,
        log_every_n_steps=50,
        precision=32)

    # start training
    if cfg.checkpoint:
        print(f"Resuming from checkpoint: {cfg.checkpoint}")
        trainer.fit(learner, ckpt_path=cfg.checkpoint)
    else:
        trainer.fit(learner)





if __name__ == '__main__':
    # cfg = parse_args()

    SEED = cfg.SEED
    if cfg.SEED == -1:
        SEED = np.random.randint(0, 1000000)
    L.seed_everything(SEED)

    try:
        outdir = cfg.OUTPUT_DIR
        print(f"Creating directory {outdir}...")
        Path(outdir).mkdir(parents=True)
    except FileExistsError:
        print(f"Directory {outdir} already exists")

    main(cfg)
