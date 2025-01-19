import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .dataset import FireDataset


class FireDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling fire dynamics data.
    """
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 3,
        transform=None,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
        drop_last: bool = False
    ):
        """
        A LightningDataModule for the FireDataset.

        Args:
            data_dir (str): Base directory that contains 'train', 'val', 'test' subfolders
            sequence_length (int): Number of time steps in each sample
            transform (callable, optional): Transform to apply on data
            batch_size (int): Batch size for each DataLoader
            num_workers (int): Number of workers for each DataLoader
            seed (int): Seed for reproducible random splitting. Default=42.
            drop_last (bool): Whether to drop the last incomplete batch in each DataLoader.
        """
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.drop_last = drop_last

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
            """
            Called only from a single GPU/TPU in distributed settings.
            Typically used to download data, tokenize, etc.
            For local data, often it's just `pass`.
            """
            pass

    def setup(self, stage=None):
        """
        Create a single dataset from 'data_dir' and split 80/10/10 for train/val/test.
        'stage' can be 'fit', 'validate', 'test', or 'predict'.
        Lightning calls:
          - setup('fit')  -> for train and val
          - setup('test') -> for test
          - setup('any')  -> if called manually
        """
        # Only do dataset instantiation once
        if not self.dataset:
            self.dataset = FireDataset(
                data_dir=self.data_dir,
                sequence_length=self.sequence_length,
                transform=self.transform
            )

            # Split lengths (rounded down by int conversion, last segment picks up any remainder)
            full_len = len(self.dataset)
            train_len = int(0.8 * full_len)
            val_len   = int(0.1 * full_len)
            test_len  = full_len - train_len - val_len  # ensures total = full_len

            # Randomly split
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(self.seed)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )
