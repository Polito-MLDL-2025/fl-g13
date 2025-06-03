#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


from pathlib import Path

import flwr
import torch
from flwr.simulation import run_simulation
from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app
from fl_g13.fl_pytorch.editing.centralized_mask import save_mask
from fl_g13.fl_pytorch.editing.centralized_mask import load_mask


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


DEBUG = False


# ## Model config

## checkpoints directory
current_path = Path.cwd()
model_save_path = current_path / f"../models/fl_dino_baseline/iid"


# Model config

## Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 12

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
device = 'cuda'

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
model.unfreeze_blocks(unfreeze_blocks)
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


# ## Compute stats for mask

from typing import Dict, Any

def compute_mask_stats(mask_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Computes various statistics for a mask represented as a dictionary
    mapping parameter names to mask tensors.

    Args:
        mask_dict: A dictionary where keys are parameter names (str)
                   and values are mask tensors (torch.Tensor) containing 0s and 1s.

    Returns:
        A dictionary containing overall and layer-wise mask statistics.
    """
    stats = {}

    # --- Overall Statistics ---
    total_elements = 0
    kept_elements_overall = 0 # Elements with value 1
    masked_elements_overall = 0 # Elements with value 0

    for name, mask_tensor in mask_dict.items():
        num_elements = mask_tensor.numel()
        kept_in_layer = torch.sum(mask_tensor == 1).item()
        masked_in_layer = num_elements - kept_in_layer

        total_elements += num_elements
        kept_elements_overall += kept_in_layer
        masked_elements_overall += masked_in_layer

        # --- Layer-wise Statistics ---
        layer_stats = {
            'num_elements': num_elements,
            'kept_elements': kept_in_layer,
            'masked_elements': masked_in_layer,
            'density': kept_in_layer / num_elements if num_elements > 0 else 0.0,
            'sparsity': masked_in_layer / num_elements if num_elements > 0 else 0.0
        }
        stats[name] = layer_stats

    # --- Add Overall Statistics to the result dictionary ---
    stats['overall'] = {
        'total_elements': total_elements,
        'kept_elements': kept_elements_overall,
        'masked_elements': masked_elements_overall,
        'density': kept_elements_overall / total_elements if total_elements > 0 else 0.0,
        'sparsity': masked_elements_overall / total_elements if total_elements > 0 else 0.0
    }

    return stats


def print_mask_stats(stats: Dict[str, Any], layer = False):
    """
    Prints the mask statistics in a readable format.

    Args:
        stats: The dictionary returned by compute_mask_stats.
    """
    if 'overall' not in stats:
        print("Invalid stats dictionary format.")
        return

    overall_stats = stats['overall']
    print("--- Overall Mask Statistics ---")
    print(f"Total Elements: {overall_stats['total_elements']}")
    print(f"Kept Elements (1s): {overall_stats['kept_elements']}")
    print(f"Masked Elements (0s): {overall_stats['masked_elements']}")
    print(f"Overall Density: {overall_stats['density']:.4f}")
    print(f"Overall Sparsity: {overall_stats['sparsity']:.4f}")
    print("-" * 30)

    if not layer:
      return

    print("--- Layer-wise Mask Statistics ---")
    # Sort layer names for consistent output
    layer_names = sorted([name for name in stats if name != 'overall'])
    for name in layer_names:
        layer_stats = stats[name]
        print(f"Layer: {name}")
        print(f"  Num Elements: {layer_stats['num_elements']}")
        print(f"  Kept Elements: {layer_stats['kept_elements']}")
        print(f"  Masked Elements: {layer_stats['masked_elements']}")
        print(f"  Density: {layer_stats['density']:.4f}")
        print(f"  Sparsity: {layer_stats['sparsity']:.4f}")
        print("-" * 20)


# ## Calculate the centralized mask

from fl_g13.fl_pytorch.editing.centralized_mask import get_centralized_mask

## config client data set params
client_partition_type = 'iid'  # 'iid' or 'shard' for non-iid dataset
client_num_partitions = 100  # equal to number of client
client_num_shards_per_partition = 10
client_batch_size = 16
client_train_test_split_ratio = 0.2
client_dataset = "cifar100"
client_seed = 42,
client_return_dataset = False,

## config get mask params
mask_model = model
mask_sparsity = 0.8
mask_type = 'global'
mask_rounds = 1
mask_func = None

## aggregate
agg_strategy = 'union'
agg_func = None

if DEBUG:
    client_num_partitions = 10
    client_batch_size = 128
    client_train_test_split_ratio = 0.9


centralized_mask = get_centralized_mask(
    client_partition_type=client_partition_type,
    client_num_partitions=client_num_partitions,
    client_num_shards_per_partition=client_num_shards_per_partition,
    client_batch_size=client_batch_size,
    client_dataset=client_dataset,
    client_seed=client_seed,
    client_return_dataset=client_return_dataset,
    mask_model=mask_model,
    mask_sparsity=mask_sparsity,
    mask_type=mask_type,
    mask_rounds=mask_rounds,
    mask_func=mask_func,
    agg_strategy=agg_strategy,
    agg_func=agg_func
)


agg_strategy = 'intersection'
centralized_mask_intersection = get_centralized_mask(
    client_partition_type=client_partition_type,
    client_num_partitions=client_num_partitions,
    client_num_shards_per_partition=client_num_shards_per_partition,
    client_batch_size=client_batch_size,
    client_dataset=client_dataset,
    client_seed=client_seed,
    client_return_dataset=client_return_dataset,
    mask_model=mask_model,
    mask_sparsity=mask_sparsity,
    mask_type=mask_type,
    mask_rounds=mask_rounds,
    mask_func=mask_func,
    agg_strategy=agg_strategy,
    agg_func=agg_func
)


compute_mask_stats(centralized_mask[1])


compute_mask_stats(centralized_mask_intersection[1])




