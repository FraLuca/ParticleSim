# from . import transform
# from .dataset_path_catalog import DatasetCatalog
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
# import torch functional
import torch.nn.functional as F


def build_transform(cfg, mode):
    if mode == "train":
        # TODO: add augmentations eventually
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.DATA_MEAN, std=cfg.INPUT.DATA_STD)
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.DATA_MEAN, std=cfg.INPUT.DATA_STD)
        ])
    return trans


def build_dataset(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    transform = build_transform(cfg, mode)
    # print(transform)

    if mode == 'train':
        dataset = CustomDataSet(csv_path=cfg.DATASETS.TRAIN, transform=transform)
    elif mode == 'val':
        dataset = CustomDataSet(csv_path=cfg.DATASETS.VAL, transform=transform)
    elif mode == 'test':
        dataset = CustomDataSet(csv_path=cfg.DATASETS.TEST, transform=transform)
    return dataset



class CustomDataSet(Dataset):
    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path).to_numpy()
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        signal = self.df[index, 0:47]
        energy = self.df[index, 47].reshape(1)
        particle_type = self.df[index, 48].reshape(1)
  
        if self.transform:
            # TODO 1: decide augmentations
            signal = self.transform(signal.reshape(1,1,-1)).squeeze(-1).squeeze(-1)

        return signal, energy, particle_type