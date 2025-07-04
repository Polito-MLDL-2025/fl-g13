import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, ShardPartitioner
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple, Dict, Any

FEDERATED_DATASET = None  # Cache the FederatedDataset

# *** -------- TRANSFORMS -------- *** #
def get_train_transforms() -> transforms.Compose:
    """
    Returns the transformations for training data.

    These transforms include resizing, random cropping, horizontal flipping,
    conversion to a tensor, and normalization using ImageNet statistics.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),  # CIFAR100 is originally 32x32
        transforms.RandomCrop(224),  # But Dino works on 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet stats
    ])
    return train_transform

def get_eval_transforms() -> transforms.Compose:
    """
    Returns the transformations for evaluation data.

    These transforms include resizing, center cropping, conversion to a tensor,
    and normalization using ImageNet statistics.
    """
    eval_transform = transforms.Compose([
        transforms.Resize(256),  # CIFAR100 is originally 32x32
        transforms.CenterCrop(224),  # But Dino works on 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet stats
    ])
    return eval_transform

def get_transforms(transform_type: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns a function that applies the specified type of transformations to a batch.

    Args:
        transform_type (str): The type of transform to apply ('train' or 'eval').

    Returns:
        Callable: A function that takes a batch and returns the transformed batch.
    """
    def apply_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
        if transform_type == 'train':
            pytorch_transforms = get_train_transforms()
        elif transform_type == 'eval':
            pytorch_transforms = get_eval_transforms()
        else:
            raise ValueError(f"No transform types with type name: {transform_type}, try one among 'train' and 'eval'")

        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        batch["fine_label"] = [int(lbl) for lbl in batch["fine_label"]]

        return batch
    return apply_transforms

# *** -------- DATALOADER -------- *** #
def reset_partition() -> None:
    """Resets the cached FederatedDataset to allow for a new one to be created."""
    global FEDERATED_DATASET
    FEDERATED_DATASET = None

def load_flwr_datasets(
    partition_id: int,
    partition_type: str,
    num_partitions: int,
    num_shards_per_partition: int,
    batch_size: int,
    dataset: str = "cifar100",
    train_test_split_ratio: float = 0.2,
    transform: Callable[[str], Callable] = get_transforms,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads and partitions a federated dataset.

    This function handles loading a dataset, partitioning it among clients using
    either IID or non-IID (shard) partitioning, and creating train/validation
    DataLoaders for a specific client partition.

    Args:
        partition_id (int): The ID of the client partition to load.
        partition_type (str): The type of partitioning ('iid' or 'shard').
        num_partitions (int): The total number of client partitions.
        num_shards_per_partition (int): The number of shards per partition for non-IID partitioning.
        batch_size (int): The batch size for the DataLoaders.
        dataset (str, optional): The name of the dataset to load. Defaults to "cifar100".
        train_test_split_ratio (float, optional): The ratio of data to use for testing. Defaults to 0.2.
        transform (Callable, optional): A function that returns the data transformations. Defaults to get_transforms.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
    """
    global FEDERATED_DATASET
    # Create the FederatedDataset only once and cache it
    if FEDERATED_DATASET is None:
        if partition_type == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif partition_type == "shard":
            partitioner = ShardPartitioner(
                num_partitions=num_partitions,
                partition_by="fine_label",
                num_shards_per_partition=num_shards_per_partition,
            )
        else:
            raise ValueError(f"Unknown partition_type: {partition_type}")

        FEDERATED_DATASET = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner},
        )

    # Load the specified client partition
    partition = FEDERATED_DATASET.load_partition(partition_id)

    # Split the partition into training and testing sets
    partition_train_test = partition.train_test_split(
        test_size=train_test_split_ratio,
        seed=seed,  # Use a seed for reproducibility
    )

    # Apply the transformations to the datasets
    partition_train_test = partition_train_test.with_transform(transform(transform_type='train'))

    # Create DataLoaders for the training and validation sets
    train_dataloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    validation_dataloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return train_dataloader, validation_dataloader

# *** -------- UTILITY FUNCTIONS -------- *** #
def collate_batch(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of data points into tensors.

    Args:
        batch (List[Dict[str, Any]]): A list of data points, where each point is a dictionary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and label tensors.
    """
    imgs = torch.stack([b["img"] for b in batch]).float()
    labels = torch.tensor([b["fine_label"] for b in batch], dtype=torch.long)
    return imgs, labels
