#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


from pathlib import Path

import flwr
import torch
from flwr.simulation import run_simulation
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets

from fl_g13.architectures import BaseDino
from fl_g13.config import RAW_DATA_DIR
from fl_g13.editing import SparseSGDM
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.datasets import get_eval_transforms
from fl_g13.fl_pytorch.server_app import get_server_app
from fl_g13.modeling.eval import eval


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()


# # Login wandb

get_ipython().system('pip install wandb')


## read .env file
import dotenv

dotenv.load_dotenv()


import wandb

# login by key in .env file
WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
wandb.login(key=WANDB_API_KEY)


# ## Build module local
# 
# Build module local such that ClientApp can use it

get_ipython().system('pip install -e ..')


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
                       "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/vision_transformer.py")
download_if_not_exists("utils.py",
                       "https://raw.githubusercontent.com/facebookresearch/dino/refs/heads/main/utils.py")


# # FL

# ## Configs

DEBUG = True


# Model config

## Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 1

## Training Hyper-parameters
batch_size = 128
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5

# FL config
K = 100
C = 0.1
J = 4
num_rounds = 30
partition_type = 'iid'

## only for partition_type = 'shard'
num_shards_per_partition = 10

## Server App config
save_every = 1
fraction_fit = C  # Sample of available clients for training
fraction_evaluate = 0.1  # Sample 50% of available clients for evaluation
min_fit_clients = 10  # Never sample less than 10 clients for training
min_evaluate_clients = 5  # Never sample less than 5 clients for evaluation
min_available_clients = 10  # Wait until all 10 clients are available
device = DEVICE
## checkpoints directory
current_path = Path.cwd()
model_save_path = current_path / f"../models/fl_dino_baseline/{partition_type}"
checkpoint_dir = model_save_path.resolve()
os.makedirs(checkpoint_dir, exist_ok=True)

## Wandb config
use_wandb = True
wandb_config = {
    # wandb param
    'name': 'FL_Dino_Baseline_iid',
    'project_name': "FL_test_chart",
    # model config param
    "fraction_fit": fraction_fit,
    "lr": lr,
    "momentum": momentum,
    'partition_type': partition_type,
    'K': K,
    'C': C,
    'J': J,
}

# model editing config
model_editing = True
mask_type = 'global'
sparsity = 0.2
mask = None

## simulation run config
NUM_CLIENTS = 100
MAX_PARALLEL_CLIENTS = 10

if DEBUG:
    use_wandb = False
    num_rounds = 2
    J = 4


# ## Define model , optimizer and loss function

from fl_g13.modeling import load_or_create

# Model
model, start_epoch = load_or_create(
        path=checkpoint_dir,
        model_class=BaseDino,
        model_config=None,
        optimizer=None,
        scheduler=None,
        device=device,
        verbose=True,
    )

model.to(DEVICE)

# optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

# Create a dummy mask for SparseSGDM
init_mask = [torch.ones_like(p, device=p.device) for p in
             model.parameters()]  # Must be done AFTER the model is moved to the device
# Optimizer, scheduler, and loss function
optimizer = SparseSGDM(
    model.parameters(),
    mask=init_mask,
    lr=lr,
    momentum=0.9,
    weight_decay=1e-5
)
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=T_max,
    eta_min=eta_min
)


# ## Define the Client, Server Apps

client = get_client_app(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=DEVICE,
    partition_type=partition_type,
    local_epochs=J,
    batch_size=batch_size,
    num_shards_per_partition=num_shards_per_partition,
    scheduler=scheduler,
    verbose=0,
    model_editing=model_editing,
    mask_type=mask_type,
    sparsity=sparsity,
    mask=mask
)


server = get_server_app(checkpoint_dir=checkpoint_dir,
                        model_class=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        scheduler=scheduler,
                        num_rounds=num_rounds,
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        min_fit_clients=min_fit_clients,
                        min_evaluate_clients=min_evaluate_clients,
                        min_available_clients=min_available_clients,
                        device=device,
                        use_wandb=use_wandb,
                        wandb_config=wandb_config,
                        save_every=save_every,
                        prefix='fl_baseline'
                        )


# ## Before training
# 
# Test model performance before fine-turning

testset = datasets.CIFAR100(RAW_DATA_DIR, train=False, download=True, transform=get_eval_transforms())
testloader = DataLoader(testset, batch_size=32)


test_loss, test_accuracy, _ = eval(testloader, model, criterion)
test_loss, test_accuracy


# ## Run the training
# 

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config["client_resources"] = {"num_cpus": 1, "num_gpus": 0.5}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# 

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config
)

