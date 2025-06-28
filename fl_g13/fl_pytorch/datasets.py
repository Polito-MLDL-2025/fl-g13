import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, ShardPartitioner
from flwr_datasets.visualization import plot_label_distributions
from torch.utils.data import DataLoader

fds = None  # Cache the FederatedDataset

# *** -------- TRANSFORMS -------- *** #

def get_train_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),  # CIFRA100 is originally 32x32
        transforms.RandomCrop(224),  # But Dino works on 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet stats
    ])

    return train_transform


def get_eval_transforms():
    eval_transform = transforms.Compose([
        transforms.Resize(256),  # CIFRA100 is originally 32x32
        transforms.CenterCrop(224),  # But Dino works on 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet stats
    ])
    return eval_transform


def get_transforms(type):
    def apply_transforms(batch):
        if type == 'train':
            pytorch_transforms = get_train_transforms()
        elif type == 'eval':
            pytorch_transforms = get_eval_transforms()
        else:
            raise ValueError(f"No transform types with type name: {type}, try one among 'train' and 'eval'")

        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        batch["fine_label"] = [int(lbl) for lbl in batch["fine_label"]]

        return batch

    return apply_transforms


# *** -------- DATALOADER -------- *** #

def reset_partition():
    global fds
    fds = None


def load_flwr_datasets(
        partition_id,
        partition_type,
        num_partitions,
        num_shards_per_partition,
        batch_size,
        dataset="cifar100",
        train_test_split_ratio=0.2,
        transform=get_transforms,
        seed=42,
):
    global fds
    if fds is None:
        if partition_type == "iid":
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={
                    "train": IidPartitioner(
                        num_partitions=num_partitions
                    ),
                },
            )
        elif partition_type == "shard":
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={
                    "train": ShardPartitioner(
                        num_partitions=num_partitions,
                        partition_by="fine_label",
                        num_shards_per_partition=num_shards_per_partition
                    )
                },
            )

    # Load data on the partition
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=train_test_split_ratio,
                                                      seed=seed)  # Using arbitrary seed for reproducibility

    # Apply transforms on the parititions
    partition_train_test = partition_train_test.with_transform(transform(type='train'))

    # Create train/val for each partition and wrap it into DataLoader
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    valloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return trainloader, valloader
def load_flwr_datasets_test(
        dataset="cifar100",
        transform=get_transforms,
        batch_size=32
):
    federated_dataset = FederatedDataset(dataset=dataset, partitioners={"test": IidPartitioner(
        num_partitions=1
    )})
    partition = federated_dataset.load_partition(0)
    partition_train_test = partition.with_transform(transform(type='eval'))

    # Create train/val for each partition and wrap it into DataLoader

    valloader = DataLoader(
        partition_train_test, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return valloader

# *** -------- UTILITY FUNCTIONS -------- *** #

def collate_batch(batch):
    imgs = torch.stack([b["img"] for b in batch]).float()
    labels = torch.tensor([b["fine_label"] for b in batch], dtype=torch.long)
    return imgs, labels
