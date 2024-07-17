import random

import lightning as L
from torch.utils.data import DataLoader, Dataset


class MixDataset(Dataset):
    def __init__(self, datasets, weights=None):
        if weights is None:
            weights = [1 / len(datasets)] * len(datasets)
        self.weights = weights
        self.datasets = datasets

    def __getitem__(self, idx):
        dataset_choice = random.choices(range(len(self.datasets)), self.weights)[0]
        chosen_dataset = self.datasets[dataset_choice]
        if idx > len(chosen_dataset):
            idx = random.randint(0, len(chosen_dataset))
        return chosen_dataset[idx]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)


class BasicDataModule(L.LightningDataModule):
    def __init__(
        self, train_dataloader_args, val_dataloader_args, train_dataset, val_dataset
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader_args = train_dataloader_args
        self.val_dataloader_args = val_dataloader_args

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_dataloader_args)
