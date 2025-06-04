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

from fl_g13.architectures import BaseDino
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app


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
# WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
WANDB_API_KEY = 'd8a0d7bc0ada694ba9c7f26bd159620f0326a74f'
wandb.login(key=WANDB_API_KEY)


# ## Build module local
# 
# Build module local such that ClientApp can use it

get_ipython().system('pip install -e ..')


# ### Download missing module for clients
# 
# Dino model,that is serialized and sent to client by server, require some modules that have to download from source code of dino model
# 

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


K = 100
C = 0.01
J = 4
num_rounds = 50
# partition_type = 'iid'
partition_type = 'shard'
Nc = 10

## only for partition_type = 'shard'
num_shards_per_partition = Nc

# checkpoint_dir = "/content/drive/MyDrive/mldl_fl/fl_dino/non_iid/checkpoints"
# ##
# os.makedirs(checkpoint_dir, exist_ok=True)


# Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 0

# Training Hyper-parameters
batch_size = 64
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5

save_every = 5
fraction_fit = C  # Sample of available clients for training
fraction_evaluate = 0.1  # Sample 50% of available clients for evaluation
min_fit_clients = 10  # Never sample less than 10 clients for training
min_evaluate_clients = 5  # Never sample less than 5 clients for evaluation
min_available_clients = 10  # Wait until all 10 clients are available
device = DEVICE
## Wandb config
use_wandb = True
wandb_config = {
    # wandb param
    'name': 'FL_Dino_Baseline_iid',
    'project_name': "FL_Dino_CIFAR100_experiment",
    # model config param
    "fraction_fit": fraction_fit,
    "lr": lr,
    "momentum": momentum,
    'partition_type': partition_type,
    'K': K,
    'C': C,
    'J': J,
}



# ### Download missing module for clients
# 
# Dino model,that is serialized and sent to client by server, require some modules that have to download from source code of dino model

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


# ## Define model , optimizer and loss function

# ## Run the training
# 

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config["client_resources"] = {"num_cpus": 1, "num_gpus": 1}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# 

from fl_g13.modeling import load_or_create
from fl_g13.editing import SparseSGDM
from torch.optim import SGD

K = 100
C = 0.1

num_rounds = 200
partition_type = 'shard'
evaluate_each = 1
# partition_type = 'iid'
NUM_CLIENTS = K
Js = [8]
Ncs = [1, 5, 10, 50]

## only for partition_type = 'shard'

## Wandb config
use_wandb = True

project_name = "FL_Dino_CIFAR100_baseline_v3"

current_path = Path.cwd()
model_save_path = current_path / f"../models/fl_dino_v3/non_iid"

for Nc in Ncs:
    for J in Js:
        print('-' * 200)
        print(f"Training Non IId model")
        print(f"Nc: {Nc}, J: {J}")
        checkpoint_dir = f"{model_save_path}/{Nc}_{J}/editing"
        print(f'checkpoint_dir:{checkpoint_dir}')
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

        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

        ## unfreeze blocks
        num_blocks = 0
        model.unfreeze_blocks(num_blocks)
        num_shards_per_partition = Nc

        ##
        os.makedirs(checkpoint_dir, exist_ok=True)

        name = f"FL_Dino_Baseline_model_non_iid_{Nc}_{J}"
        
        wandb_config = {
            # wandb param
            'name': name,
            'project_name': project_name,
            'run_id': f"{name}",
            # model config param
            "fraction_fit": fraction_fit,
            "lr": lr,
            "momentum": momentum,
            'partition_type': partition_type,
            'K': K,
            'C': C,
            'J': J,
            'Nc': Nc
        }
        # model editing config
        model_editing = False
        mask_type = 'global'
        sparsity = 0.8
        mask = None
        model_editing_batch_size = 16
        if model_editing:
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

        client = get_client_app(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            partition_type=partition_type,
            local_epochs=1,
            local_steps=J,
            batch_size=batch_size,
            num_shards_per_partition=num_shards_per_partition,
            scheduler=scheduler,
            verbose=0,
            model_editing=model_editing,
            mask_type=mask_type,
            sparsity=sparsity,
            mask=mask,
            model_editing_batch_size=model_editing_batch_size,
            mask_func=None
        )
        compute_round = num_rounds + 1 - start_epoch
        server = get_server_app(checkpoint_dir=checkpoint_dir,
                                model_class=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                scheduler=scheduler,
                                num_rounds=compute_round,
                                fraction_fit=fraction_fit,
                                fraction_evaluate=fraction_evaluate,
                                min_fit_clients=min_fit_clients,
                                min_evaluate_clients=min_evaluate_clients,
                                min_available_clients=min_available_clients,
                                device=device,
                                use_wandb=use_wandb,
                                wandb_config=wandb_config,
                                save_every=save_every,
                                prefix='fl_baseline',
                                evaluate_each=evaluate_each
                                )
        # Run simulation
        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=NUM_CLIENTS,
            backend_config=backend_config
        )




