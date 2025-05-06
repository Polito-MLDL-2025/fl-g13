#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' pip install -e ..')


from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app
from fl_g13.fl_pytorch.model import get_experiment_setting, Net
from fl_g13.architectures import BaseDino
from flwr.simulation import run_simulation
from fl_g13.fl_pytorch.constants import (
    NUM_CLIENTS, 
    DEFAULT_FRACTION_FIT, 
    DEFAULT_NUM_ROUNDS, 
    DEFAULT_LOCAL_EPOCHS, 
    BATCH_SIZE,
    NUM_SHARDS_PER_PARTITION,
    DEFAULT_FRACTION_EVALUATE,
    MIN_NUM_CLIENTS
)


DEBUG = True

number_of_rounds = DEFAULT_NUM_ROUNDS
fraction_fit = DEFAULT_FRACTION_FIT
fraction_evaluate = DEFAULT_FRACTION_EVALUATE
number_of_clients = NUM_CLIENTS
min_num_clients = MIN_NUM_CLIENTS
show_distribution = False
local_epochs = DEFAULT_LOCAL_EPOCHS
batch_size = BATCH_SIZE
num_shards_per_partition = NUM_SHARDS_PER_PARTITION
use_wandb = True
model_class = BaseDino

if DEBUG:
    number_of_rounds = 2
    fraction_fit = 1
    number_of_clients = 3
    min_num_clients = 3
    show_distribution = True
    local_epochs = 2
    batch_size = 128
    num_shards_per_partition = 6
    use_wandb = False
    model_class = BaseDino


checkpoint_dir = "./../models/"


from flwr_datasets import FederatedDataset, partitioner
from fl_g13.fl_pytorch.datasets import show_partition_distribution
    
if show_distribution:
    fds = FederatedDataset(
            dataset="cifar100",
            partitioners={"train": partitioner.IidPartitioner(num_partitions=number_of_clients)}
        )
    p = fds.partitioners["train"]
    show_partition_distribution(p)


starting_lr = 0.001
partition_type = "iid" # or "shard"
momentum = 0.9
wandb_config = {
    "partition_type": partition_type,
    "starting_lr": starting_lr,
    "momentum": momentum,
}


model, optimizer, criterion, device, scheduler = get_experiment_setting(checkpoint_dir, model_class, starting_lr, momentum)
client_app = get_client_app(
    model=model, 
    optimizer=optimizer, 
    criterion=criterion, 
    device=device, 
    partition_type=partition_type, 
    local_epochs=local_epochs,
    batch_size=batch_size,
    num_shards_per_partition=num_shards_per_partition,
    scheduler=scheduler,
)
server_app = get_server_app(
    model_class=model_class, 
    optimizer=optimizer, 
    criterion=criterion, 
    device=device, 
    num_rounds=number_of_rounds, 
    min_available_clients=number_of_clients,
    min_fit_clients=min_num_clients,
    min_evaluate_clients=min_num_clients,
    checkpoint_dir=checkpoint_dir,
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    use_wandb=use_wandb,
    wandb_config=wandb_config,
    scheduler=scheduler,
)


import os
import urllib.request


def download_if_not_exists(file_path: str, file_url: str):
    """
    Checks if a file exists at the given path. If it does not, downloads it from the specified URL.

    Parameters:
    - file_path (str): The local path to check and save the file.
    - file_url (str): The URL from which to download the file.
    """
    if not os.path.exists(file_path):
        print(f"'{file_path}' not found. Downloading from {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download file: {e}")
    else:
        print(f"'{file_path}' already exists.")


download_if_not_exists("vision_transformer.py",
                       "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/vision_transformer.py")
download_if_not_exists("utils.py",
                       "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/utils.py")


run_simulation(
    client_app=client_app,
    server_app=server_app,
    num_supernodes=number_of_clients
)


from fl_g13.fl_pytorch.datasets import plot_results

strategy = server_app._strategy
results = strategy.results
if results:
    print("Contenuto di results.json:", results)

plot_results(results)




