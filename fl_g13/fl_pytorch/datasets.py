import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, ShardPartitioner
from flwr_datasets.visualization import plot_label_distributions
from torch.utils.data import DataLoader

fds = None  # Cache the FederatedDataset


def get_eval_transforms():
    eval_transform = transforms.Compose([
        transforms.Resize(256),  # CIFRA100 is originally 32x32
        transforms.CenterCrop(224),  # But Dino works on 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return eval_transform


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

def load_flwr_datasets(
        partition_id,
        partition_type,
        num_partitions,
        num_shards_per_partition,
        batch_size,
        dataset="cifar100",
        train_test_split_ratio=0.2,
        transform=get_transforms,
        seed=42
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


# *** -------- UTILITY FUNCTIONS -------- *** #

def collate_batch(batch):
    imgs = torch.stack([b["img"] for b in batch]).float()
    labels = torch.tensor([b["fine_label"] for b in batch], dtype=torch.long)
    return imgs, labels


# ! *** -------- PLOTTERS (TODO) -------- *** #

def show_partition_distribution(partitioner):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 15))

    plot_label_distributions(
        partitioner,
        label_name="fine_label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
        axis=axes[0][0]
    )

    plot_label_distributions(
        partitioner,
        label_name="fine_label",
        plot_type="bar",
        size_unit="percent",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
        axis=axes[0][1]
    )

    plot_label_distributions(
        partitioner,
        label_name="fine_label",
        plot_type="heatmap",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
        axis=axes[1][0]
    )

    plot_label_distributions(
        partitioner,
        label_name="fine_label",
        plot_type="heatmap",
        size_unit="percent",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
        axis=axes[1][1]
    )

    # show the plot even if the script is run in a notebook
    plt.show()
    plt.close(fig)


def plot_results(results):
    # Extract data for plotting
    rounds = [entry['round'] for entry in results['federated_evaluate']]
    centralized_rounds = [entry['round'] for entry in results['centralized_evaluate']]
    federated_loss = [entry['federated_evaluate_loss'] for entry in results['federated_evaluate']]
    federated_accuracy = [entry['federated_evaluate_accuracy'] for entry in results['federated_evaluate']]
    centralized_loss = [entry['centralized_loss'] for entry in results['centralized_evaluate']]
    centralized_accuracy = [entry['centralized_accuracy'] for entry in results['centralized_evaluate']]
    avg_drift = [entry['avg_drift'] for entry in results['client_fit']]

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot federated_evaluate_loss
    axes[0].plot(rounds, federated_loss, label='Federated Evaluate Loss', marker='o', color='red')

    # Plot federated_evaluate_accuracy
    axes[0].plot(rounds, federated_accuracy, label='Federated Evaluate Accuracy', marker='o', color='blue')

    # Plot centralized_loss
    axes[0].plot(centralized_rounds, centralized_loss, label='Centralized Loss', marker='o', color='orange')

    # Plot federated_evaluate_accuracy
    axes[0].plot(centralized_rounds, centralized_accuracy, label='Centralized Accuracy', marker='o', color='green')

    axes[1].plot(rounds, avg_drift, label='Average Drift', marker='o', color='pink')

    axes[0].set_title('Andamento di Federated/Centralized Loss e Accuracy')
    axes[1].set_title('Andamento Average Drift')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss and Accuracy')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Drift')
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True)
    axes[1].grid(True)

    plt.show()
