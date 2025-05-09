#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch
from fl_g13.editing.sparseSGDM import SparseSGDM
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

import flwr
from flwr.simulation import run_simulation
from fl_g13.architectures import BaseDino
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app

print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")


import os
import urllib.request

# TODO: move in a "make client dependencies function"
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

download_if_not_exists(
    "vision_transformer.py",
    "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/vision_transformer.py"
)

download_if_not_exists(
    "utils.py",
    "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/utils.py"
)


# Settings
CHECKPOINT_DIR = "/home/massimiliano/Projects/fl-g13/checkpoints"

# Model hyper-parameters
model_class = BaseDino
head_layers=3
head_hidden_size=512
dropout_rate=0.0
unfreeze_blocks=1

# Training hyper-parameters
starting_lr = 1e-3
momentum = 0.9
weight_decay=1e-5
T_max=8
eta_min=1e-5

# Federated Training setting
batch_size = 64
local_epochs = 2
number_of_rounds = 2
fraction_fit = 1
fraction_evaluate = 0.1
number_of_clients = 3
min_num_clients = 3
partition_type = "iid" # or "shard"
num_shards_per_partition = 6
use_wandb = False

# Device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
backend_config = {
    "client_resources": {
        "num_cpus": 1, 
        "num_gpus": 0
    }
}

# When running on GPU, assign an entire GPU for each client
if device == "cuda":
    backend_config["client_resources"] = {"num_cpus": 1, "num_gpus": 1}

    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

print(f"Training on {device}")


# Model
model = model_class(
    head_layers=head_layers, 
    head_hidden_size=head_hidden_size, 
    dropout_rate=dropout_rate, 
    unfreeze_blocks=unfreeze_blocks
    )
model.to(device)

mask = [torch.ones_like(p, device=p.device) for p in model.parameters()] # Must be done AFTER the model is moved to CUDA
optimizer = SparseSGDM(
    model.parameters(),
    mask=mask,
    lr=starting_lr,
    momentum=momentum,
    weight_decay=weight_decay
    )
scheduler = CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=T_max, 
    eta_min=eta_min
    )
criterion = CrossEntropyLoss()

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
    model_config=model.get_config(), 
    optimizer=optimizer, 
    criterion=criterion, 
    device=device, 
    num_rounds=number_of_rounds, 
    min_available_clients=number_of_clients,
    min_fit_clients=min_num_clients,
    min_evaluate_clients=min_num_clients,
    checkpoint_dir=CHECKPOINT_DIR,
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    #use_wandb=use_wandb,
    #wandb_config=wandb_config,
    scheduler=scheduler,
)


run_simulation(
    client_app=client_app,
    server_app=server_app,
    num_supernodes=number_of_clients,
    backend_config=backend_config,
)




