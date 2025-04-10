import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from loguru import logger
from tqdm import tqdm
import typer

from fl_g13.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Downloading dataset cifar100...")
    train_dataset = load_cifar100(data_dir=RAW_DATA_DIR, train=True)
    test_dataset = load_cifar100(data_dir=RAW_DATA_DIR, train=False)
    logger.success("Downloading dataset complete.")
    # logger.info("Processing dataset...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Processing dataset complete.")
    # -----------------------------------------

def load_cifar100(data_dir="data/raw", train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    print(f"{'Training' if train else 'Test'} set downloaded to: {data_dir}")
    return dataset

def check_distribution(dataset, name="Dataset"):
    labels = [dataset[i][1] for i in range(len(dataset))]
    dist = Counter(labels)
    print(f"{name} distribution (class: count):")
    print(dict(sorted(dist.items())))


class SubsetToDataset(Dataset):
    def __init__(self, subset):
        self.data = [subset[i][0] for i in range(len(subset))]
        self.targets = [subset[i][1] for i in range(len(subset))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
def train_test_split(dataset, train_ratio=0.8,random_state=42):
    targets = np.array(dataset.targets)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx,test_idx = next(sss.split(targets, targets))
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    return train_dataset,test_dataset


def check_subset_distribution(subset, plot=False):
    labels = []

    # Grab labels from the subset
    for i in range(len(subset)):
        _, label = subset[i]
        labels.append(label)

    label_count = Counter(labels)

    return label_count
def iid_sharding(dataset, k_clients, seed=42):
    np.random.seed(seed)

    # If the input is a Subset, unwrap its indices
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        base_indices = dataset.indices
    else:
        base_dataset = dataset
        base_indices = list(range(len(dataset)))

    # Shuffle and split
    data_per_client = len(base_indices) // k_clients
    shuffled_indices = np.random.permutation(base_indices)

    client_subsets = [
        Subset(base_dataset, shuffled_indices[i * data_per_client: (i + 1) * data_per_client])
        for i in range(k_clients)
    ]

    return client_subsets

def filter_subset_keep_classes(subset, keep_classes=[]):
    filtered_indices = [i for i in range(len(subset)) if subset[i][1] in keep_classes]
    return Subset(subset, filtered_indices)

def non_iid_sharding(dataset, k_clients,keep_classes=None, keep_random=1, seed=42):

    unique_targets = set(dataset.targets)

    if keep_classes is None:
        random.seed(seed)
        keep_classes = random.sample(unique_targets,keep_random)

    cut_dataset = filter_subset_keep_classes(dataset, keep_classes)
    return iid_sharding(cut_dataset, k_clients)


if __name__ == "__main__":
    app()
