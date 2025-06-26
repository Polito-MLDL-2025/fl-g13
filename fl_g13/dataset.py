from collections import Counter
from pathlib import Path
import random

from loguru import logger
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import typer

from fl_g13.config import RAW_DATA_DIR

app = typer.Typer()

# TODO add changes made by thanh


def check_distribution(dataset, name="Dataset"):
    labels = [dataset[i][1] for i in range(len(dataset))]
    dist = Counter(labels)
    print(f"{name} distribution (class: count):")
    print(dict(sorted(dist.items())))


# TODO remove random state default parameter
def train_test_split(dataset, train_ratio=0.8, random_state=42):
    targets = np.array(dataset.targets)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, test_idx = next(sss.split(targets, targets))
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    return train_dataset, test_dataset


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
        Subset(base_dataset, shuffled_indices[i * data_per_client : (i + 1) * data_per_client])
        for i in range(k_clients)
    ]

    return client_subsets


def filter_subset_keep_classes(subset, keep_classes=[]):
    filtered_indices = [i for i in range(len(subset)) if subset[i][1] in keep_classes]
    return Subset(subset, filtered_indices)


def non_iid_sharding(dataset, k_clients, keep_classes=None, keep_random=1, seed=42):
    unique_targets = list(set(dataset.targets))

    if keep_classes is None:
        random.seed(seed)
        keep_classes = random.sample(unique_targets, keep_random)

    cut_dataset = filter_subset_keep_classes(dataset, keep_classes)
    return iid_sharding(cut_dataset, k_clients)


def load_cifar100(data_dir="data/raw", train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)

    print(f"{'Training' if train else 'Test'} set downloaded to: {data_dir}")
    return dataset


def update_dataloader(dataloader, new_batch_size,shuffle=False):
    return DataLoader(
        dataset=dataloader.dataset,
        batch_size=new_batch_size,
        shuffle=shuffle,
        sampler=dataloader.sampler,
        prefetch_factor=dataloader.prefetch_factor,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        in_order=dataloader.in_order,
        persistent_workers=dataloader.persistent_workers
    )

@app.command()
def main(
    path: Path = RAW_DATA_DIR,
):
    logger.info("Downloading CIFAR100 dataset into the RAW_DATA_DIR...")

    # Download training and test sets into the specified RAW_DATA_DIR
    load_cifar100(data_dir=str(path), train=True)
    load_cifar100(data_dir=str(path), train=False)

    logger.success("✅ Downloading CIFAR100 dataset complete.")


if __name__ == "__main__":
    app()
