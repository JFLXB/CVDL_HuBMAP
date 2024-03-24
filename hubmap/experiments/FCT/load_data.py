from typing import Tuple, Callable
from torch.utils.data import random_split, DataLoader

from hubmap.data import DATA_DIR
from hubmap.dataset import DatasetFromSubset
from hubmap.dataset import BaseDataset, ExpertDataset
from hubmap.dataset.transforms import Compose


def make_expert_loader(
    train_transformations: Compose,
    test_transformations: Compose,
    train_ratio: float = 0.8,
) -> Callable[[int], Tuple[DataLoader, DataLoader]]:
    dataset = ExpertDataset(DATA_DIR)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset_sub, test_dataset_sub = random_split(dataset, [train_size, test_size])
    train_dataset = DatasetFromSubset(
        train_dataset_sub, transform=train_transformations
    )
    test_dataset = DatasetFromSubset(test_dataset_sub, transform=test_transformations)

    def _load(batch_size: int) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    return _load


def make_annotated_loader(
    train_transformations: Compose,
    test_transformations: Compose,
    train_ratio: float = 0.8,
    with_background: bool = False,
) -> Callable[[int], Tuple[DataLoader, DataLoader]]:
    dataset = BaseDataset(DATA_DIR, with_background=with_background)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset_sub, test_dataset_sub = random_split(dataset, [train_size, test_size])
    train_dataset = DatasetFromSubset(
        train_dataset_sub, transform=train_transformations
    )
    test_dataset = DatasetFromSubset(test_dataset_sub, transform=test_transformations)

    def _load(batch_size: int) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    return _load
