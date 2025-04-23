from flwr_datasets import FederatedDataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from flwr_datasets.partitioner import IidPartitioner, ShardPartitioner
from flwr_datasets.visualization import plot_label_distributions
import matplotlib.pyplot as plt
from torch import stack, tensor, long
import numpy as np

BATCH_SIZE, NUM_SHARDS_PER_PARTITION = 32, 2

def my_collate(batch):                   # batch è list(dict) con tensor già trasformati
    imgs  = stack([b["img"] for b in batch]).float()
    labels= tensor([b["fine_label"] for b in batch], dtype=long)
    return imgs, labels

def get_eval_transforms():
    eval_transform = transforms.Compose([
        transforms.Resize(256), # CIFRA100 is originally 32x32
        transforms.CenterCrop(224), # But Dino works on 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    ])
    return eval_transform

def get_transforms():
    """Return a function that apply standard transformations to images."""

    def apply_transforms(batch):
        # Applica le trasformazioni alle immagini
        pytorch_transforms = get_eval_transforms()
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        batch["fine_label"] = [int(lbl) for lbl in batch["fine_label"]]
        
        return batch

    return apply_transforms

fds = None # Cache the FederatedDataset

def load_datasets(partition_id: int, num_partitions: int, partitionType: str="iid"):

    global fds
    if fds is None:
        if partitionType == "iid":
            fds = FederatedDataset(
                dataset="cifar100",
                partitioners={
                    "train": IidPartitioner(num_partitions=num_partitions),
                },
            )
        elif partitionType == "shard":
            fds = FederatedDataset(
                dataset="cifar100",
                partitioners={
                    "train": ShardPartitioner(
                        num_partitions=num_partitions, partition_by="fine_label", num_shards_per_partition=NUM_SHARDS_PER_PARTITION
                    )
                },
            )

    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE, collate_fn=my_collate) # local validation set partition loader for each client
    return trainloader, valloader


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
    # Estrai i dati per il plot
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

    # Configura il grafico
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

    # Mostra il grafico
    plt.show()