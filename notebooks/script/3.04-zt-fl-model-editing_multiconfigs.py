#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import os
import urllib.request
from itertools import product
from pathlib import Path

import torch
from flwr.simulation import run_simulation
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.datasets import reset_partition
from fl_g13.fl_pytorch.server_app import get_server_app
from fl_g13.modeling import load_or_create


# # Login wandb

get_ipython().system('pip install wandb --quiet')


## read .env file
import dotenv

dotenv.load_dotenv()


import wandb

# login by key in .env file
WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
# WANDB_API_KEY = 'd8a0d7bc0ada694ba9c7f26bd159620f0326a74f'
wandb.login(key=WANDB_API_KEY)


# ## Build module local
# 
# Build module local such that ClientApp can use it

get_ipython().system('pip install -e .. --quiet')


# ### Download missing module for clients
# 
# Dino model,that is serialized and sent to client by server, require some modules that have to download from source code of dino model
# 

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

# ----------------------------------------
# Device Setup
# ----------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")

# ----------------------------------------
# Client Resource Configuration
# ----------------------------------------
backend_config = {
    "client_resources": {
        "num_cpus": 1,
        "num_gpus": 1 if DEVICE == "cuda" else 0.0
    }
}

# ----------------------------------------
# Experiment Metadata
# ----------------------------------------
project_name = "FL_Dino_CIFAR100_masking_grid_search_v4"
partition_type = 'shard'  # 'iid' or 'shard'
partition_name = 'iid' if partition_type == 'iid' else 'non-iid'
model_editing = True
use_wandb = True

# ----------------------------------------
# Paths and Checkpoints
# ----------------------------------------
previous_model_path = '../models/fl_baseline/fl_baseline_model/fl_fl_baseline_BaseDino_epoch_200_noniid_1_8.pth'
current_path = Path.cwd()
model_save_path = current_path / "../models/fl_dino_v4/non_iid"

# ----------------------------------------
# Global Experiment Settings
# ----------------------------------------
K = 100  # Total clients
C = 0.1  # Client sampling fraction
NUM_CLIENTS = K
num_rounds = 55
evaluate_each = 2

# Evaluation thresholds
fraction_fit = C
fraction_evaluate = 0.1
min_fit_clients = 10
min_evaluate_clients = 5
min_available_clients = 10

# ----------------------------------------
# Model & Training Hyperparameters
# ----------------------------------------
batch_size = 64
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5
save_every = 5
num_blocks = 12
device = DEVICE

model_config = {
    "head_layers": 3,
    "head_hidden_size": 512,
    "dropout_rate": 0.0,
    "unfreeze_blocks": 0
}

# --------------------------
# Main Experiment Loop
# --------------------------
Js = [8]
Ncs = [1, 5, 10, 50]
mask_calibration_round_s = [1]  # E.g. [1, 3]
mask_types = ['local']  # E.g. ['global', 'local']
sparsitys = [0.8, 0.7]
model_editing_batch_size = 1
mask = None  # Default mask placeholder

# ----------------------------------------
# Grid Search Execution Loop
# ---------------------------------------


# ## Run the training
# 

# 

for J, Nc, mask_calibration_round, mask_type, sparsity in product(Js, Ncs, mask_calibration_round_s, mask_types,
                                                                  sparsitys):
    reset_partition()
    print('-' * 100)
    print(
        f"Training configuration: J={J}, Nc={Nc}, mask_round={mask_calibration_round}, type={mask_type}, sparsity={sparsity}")

    checkpoint_dir = f"{model_save_path}/{Nc}_{J}_{mask_type}_{mask_calibration_round}_{sparsity}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load or create model
    model, start_epoch = load_or_create(
        path=previous_model_path,
        model_class=BaseDino,
        model_config=model_config,
        optimizer=None,
        scheduler=None,
        device=DEVICE,
        verbose=True
    )
    model.to(DEVICE)
    model.unfreeze_blocks(num_blocks)

    # Set up optimizer and scheduler
    if model_editing:
        init_mask = [torch.ones_like(p, device=p.device) for p in model.parameters()]
        optimizer = SparseSGDM(
            model.parameters(),
            mask=init_mask,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    criterion = CrossEntropyLoss()

    # Construct WandB run config
    run_name = f"FL_Dino_Baseline_model_{partition_name}_{J}_{mask_type}_{mask_calibration_round}_{sparsity}"
    wandb_config = {
        'name': run_name,
        'project_name': project_name,
        'run_id': run_name,
        'fraction_fit': fraction_fit,
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'partition_type': partition_type,
        'K': K,
        'C': C,
        'J': J,
        'Nc': Nc,
        'mask_calibration_round': mask_calibration_round,
        'mask_type': mask_type,
        'sparsity': sparsity,
        'T_max': T_max,
        'eta_min': eta_min,
        'unfreeze_blocks': num_blocks
    }

    # Client Setup
    client = get_client_app(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        partition_type=partition_type,
        local_epochs=1,
        local_steps=J,
        batch_size=batch_size,
        num_shards_per_partition=Nc,
        scheduler=scheduler,
        verbose=0,
        model_editing=model_editing,
        mask_type=mask_type,
        sparsity=sparsity,
        mask=mask,
        model_editing_batch_size=model_editing_batch_size,
        mask_func=None,
        mask_calibration_round=mask_calibration_round
    )

    # Server Setup
    compute_rounds = num_rounds + 1 - start_epoch
    server = get_server_app(
        checkpoint_dir=checkpoint_dir,
        model_class=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_rounds=compute_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        device=DEVICE,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
        save_every=save_every,
        prefix='fl_baseline',
        evaluate_each=evaluate_each,
        model=model,
        start_epoch=start_epoch
    )

    # Run Simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config
    )

    wandb.finish()




