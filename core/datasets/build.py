# from . import transform
# from .dataset_path_catalog import DatasetCatalog
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
# import torch functional
import torch.nn.functional as F


def build_transform(cfg, mode, var):
    if var == 'input':
        if mode == "train":
            # TODO: add augmentations eventually
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.INPUT.DATA_MEAN[:47], std=cfg.INPUT.DATA_STD[:47])
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.INPUT.DATA_MEAN[:47], std=cfg.INPUT.DATA_STD[:47])
            ])
    elif var == 'target':
        if mode == "train":
            # TODO: add augmentations eventually
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.INPUT.DATA_MEAN[47:49], std=cfg.INPUT.DATA_STD[47:49])
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.INPUT.DATA_MEAN[47:49], std=cfg.INPUT.DATA_STD[47:49])
            ])

    return trans


def build_dataset(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    transform_input = build_transform(cfg, mode, 'input')
    transform_target = build_transform(cfg, mode, 'target')
    # print(transform)

    if mode == 'train':
        dataset = CustomDataSet(csv_path=cfg.DATASETS.TRAIN, transform_input=transform_input, transform_target=transform_target)
    elif mode == 'val':
        dataset = CustomDataSet(csv_path=cfg.DATASETS.VAL, transform_input=transform_input, transform_target=transform_target)
    elif mode == 'test':
        dataset = CustomDataSet(csv_path=cfg.DATASETS.TEST, transform_input=transform_input, transform_target=transform_target)
    return dataset



class CustomDataSet(Dataset):
    def __init__(self, csv_path, transform_input=None, transform_target=None):

        self.df = pd.read_csv(csv_path).to_numpy()
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        signal = self.df[index, 0:47]
        energy = self.df[index, 47:49]
        particle_type = self.df[index, 49].reshape(1)
  
        if self.transform_input:
            # TODO 1: decide augmentations
            signal = self.transform_input(signal.reshape(1,1,-1)).squeeze(-1).squeeze(-1)
        if self.transform_target:
            # TODO 1: decide augmentations
            energy = self.transform_target(energy.reshape(1,1,-1)).squeeze(-1).squeeze(-1)

        return signal, energy, particle_type