from . import transform
# from .dataset_path_catalog import DatasetCatalog
import pandas as pd


def build_transform(cfg, mode):
    if mode == "train":
        # TODO: add augmentations eventually
        trans = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.DATA_MEAN, std=cfg.INPUT.DATA_STD)
        ])
    else:
        trans = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.DATA_MEAN, std=cfg.INPUT.DATA_STD)
        ])
    return trans


def build_dataset(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    # transform = build_transform(cfg, mode)
    # print(transform)
    transform = None

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
        energy = self.df[index, 47]
        particle_type = self.df[index, 48]
  
        if self.transform:
            # TODO 1: decide augmentations
            # TODO 2: normalize data per channel? 
            signal = self.transform(signal)

        return signal, energy, particle_type