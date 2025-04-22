#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


from pathlib import Path

import flwr
import torch
from flwr.common import Context
from flwr.simulation import run_simulation
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, Normalize, ToTensor

from fl_g13 import dataset as dataset_handler
from fl_g13.architectures import BaseDino
from fl_g13.config import RAW_DATA_DIR
from fl_g13.dataset import train_test_split
from fl_g13.fl_pytorch.server_app import get_server_app


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()


# # Load data

# Define preprocessing pipeline
train_transform = Compose([
    Resize(256),  # CIFRA100 is originally 32x32
    RandomCrop(224),  # But Dino works on 224x224
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

eval_transform = Compose([
    Resize(256),  # CIFRA100 is originally 32x32
    CenterCrop(224),  # But Dino works on 224x224
    ToTensor(),
    Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=train_transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=eval_transform)

train_dataset, val_dataset = train_test_split(cifar100_train, 0.8, random_state=None)
test_dataset = cifar100_test

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# Dataloaders
BATCH_SIZE = 128
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# I.I.D Sharding Split
## k client
k = 10
clients_dataset_train = dataset_handler.iid_sharding(train_dataset, k)
clients_dataset_val = dataset_handler.iid_sharding(val_dataset, k)


clients_dataloader_train = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True) for d in clients_dataset_train]
clients_dataloader_val = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True) for d in clients_dataset_val]


# ## Model

# ## Init model , optimizer and loss function

# Hyper-parameters
LR = 1e-2

# Model
model = BaseDino()
model.to(DEVICE)
print(f"Model: {model}")

# Optimizer, scheduler, and loss function
optimizer = SGD(model.parameters(), lr=LR)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=8,  # First restart after 8 epochs
    T_mult=2,  # Double the interval between restarts each time
    eta_min=1e-5  # Minimum learning rate after annealing
)
criterion = CrossEntropyLoss()


# # Define the ClientApp

# ## Build module local
# 
# Build module local such that ClientApp can use it

get_ipython().system('pip install -e ..')


# ## create FlowerClient instances  

'''
Function load data client is to simulate the distribution data into each client
In the real case, each client will have its dataset
'''


def load_data_client(context: Context):
    partition_id = context.node_config["partition-id"]
    print(f"Client {partition_id} is ready to train")
    return clients_dataloader_train[partition_id], clients_dataloader_val[partition_id]


# ### Create instant of ClientApp

from fl_g13.fl_pytorch.client_app import get_client_app

config = {'local-epochs': 3}

client = get_client_app(load_data_client,
                        model=model, optimizer=optimizer, criterion=criterion,
                        device=DEVICE, config=config)


# # Define the Flower ServerApp
# 
# Customize built-in strategy Federated Averaging (FedAvg) of Flower to combine hyperparams in server-side and save model for each k epoch
# 
# The strategy could also incremental training 

# ## Create instant of ServerApp

def get_datatest_fn(context: Context):
    return test_dataloader


## checkpoints directory
current_path = Path.cwd()
model_test_path = current_path / "../models/fl_baseline"
model_test_path.resolve()

num_rounds = 2
save_every = 1
fraction_fit = 1.0  # Sample 100% of available clients for training
fraction_evaluate = 0.5  # Sample 50% of available clients for evaluation
min_fit_clients = 10  # Never sample less than 10 clients for training
min_evaluate_clients = 5  # Never sample less than 5 clients for evaluation
min_available_clients = 10  # Wait until all 10 clients are available
device = DEVICE
use_wandb = False

server = get_server_app(checkpoint_dir=model_test_path.resolve(),
                        model_class=BaseDino,
                        optimizer=optimizer,
                        criterion=criterion,
                        scheduler=scheduler,
                        get_datatest_fn=get_datatest_fn,
                        num_rounds=num_rounds,
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        min_fit_clients=min_fit_clients,
                        min_evaluate_clients=min_evaluate_clients,
                        min_available_clients=min_available_clients,
                        device=device,
                        use_wandb=use_wandb,
                        save_every=save_every
                        )


# # Run the training
# 

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config["client_resources"] = {"num_cpus": 1, "num_gpus": 1}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# ### Download missing module for clients
# 
# Dino model,that is serialized and sent to client by server, require some modules that have to download from source code of dino model
# 

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
                       "wget https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/vision_transformer.py")
download_if_not_exists("utils.py",
                       "wget https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/utils.py")


# 

NUM_CLIENTS = 10


# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)




