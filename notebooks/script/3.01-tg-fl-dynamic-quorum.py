#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import flwr
import torch

from fl_g13.fl_pytorch.editing.centralized_mask import load_mask
from fl_g13.fl_pytorch.editing.centralized_mask import save_mask
from fl_g13.fl_pytorch import build_fl_dependencies


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

build_fl_dependencies()


import dotenv

dotenv.load_dotenv()


import wandb

# login by key in .env file
WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
wandb.login(key=WANDB_API_KEY)


# # FL CONFIG

from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM

DEBUG = False


import os

CHECKPOINT_DIR = "/Users/ciovi/Desktop/coding/mldl/fl-g13/checkpoints"
name = 'dynamic-quorum'

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
J = 8
num_rounds = 30
partition_type = 'shard'

## only for partition_type = 'shard'
num_shards_per_partition = 1

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
model_save_path = current_path / f"../models/fl_dino_baseline/noniid"
checkpoint_dir = model_save_path.resolve()
os.makedirs(checkpoint_dir, exist_ok=True)

## Wandb config
use_wandb = True
name = "Test_Dynamic_Quorum_v2"
strategy = 'quorum'

wandb_config = {
    # wandb param
    'name': name,
    'project_name': "testing-quorum",
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
mask_type = 'local'
sparsity = 0.8
mask = None

## simulation run config
NUM_CLIENTS = 100
MAX_PARALLEL_CLIENTS = 10

if DEBUG:
    use_wandb = False
    num_rounds = 2
    J = 4


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

unfreeze_blocks = 12
model.unfreeze_blocks(unfreeze_blocks)
# optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

# Create a dummy mask for SparseSGDM
# Must be done AFTER the model is moved to the device
init_mask = [torch.ones_like(p, device=p.device) for p in model.parameters()]

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
    kept_elements_overall = 0  # Elements with value 1
    masked_elements_overall = 0  # Elements with value 0

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

def print_mask_stats(stats: Dict[str, Any], layer=False):
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


from fl_g13.fl_pytorch.editing.centralized_mask import get_client_masks
from fl_g13.fl_pytorch.editing.centralized_mask import aggregate_by_sum
from fl_g13.editing.masking import mask_dict_to_list

## config client data set params
client_partition_type = 'shard'  # 'iid' or 'shard' for non-iid dataset
client_num_partitions = 100  # equal to number of client
client_num_shards_per_partition = 1
client_batch_size = 16

## config get mask params
mask_model = model
mask_sparsity = 0.8
mask_type = 'local'
mask_rounds = 3

return_scores = False

file_name = f'{num_shards_per_partition}_{J}_sum_mask_{mask_type}_{mask_sparsity}_{mask_rounds}.pth'

if os.path.isfile(CHECKPOINT_DIR + f'/masks/{file_name}'):
    print('Loaded File from memory')
    sum_mask = mask_dict_to_list(model, load_mask(CHECKPOINT_DIR + f'/masks/{file_name}'))
else:
    print(f"No mask checkpoint found ({file_name}). Computing sum mask")
    masks, scores, _ = get_client_masks(
        ## config client data set params
        client_partition_type=client_partition_type,  # 'iid' or 'shard' for non-iid dataset
        client_num_partitions=client_num_partitions,  # equal to number of client
        client_num_shards_per_partition=client_num_shards_per_partition,
        client_batch_size=client_batch_size,

        ## config get mask params
        mask_model=mask_model,
        mask_sparsity=mask_sparsity,
        mask_type=mask_type,
        mask_rounds=mask_rounds,
    )
    sum_mask = aggregate_by_sum(masks)
    save_mask(sum_mask, CHECKPOINT_DIR + f'/masks/{file_name}')
    sum_mask = mask_dict_to_list(model, sum_mask)


# # Simulation

num_rounds = 220

evaluate_each = 1
partition_type = 'shard' # or 'shard' for non iid
J = 8
Ncs = 1

model_editing_batch_size=16
mask=None

num_blocks = 12
previous_model_path = CHECKPOINT_DIR + '/fl_dino_v4/non_iid/1_8/fl_fl_baseline_BaseDino_epoch_200_noniid_1_8.pth'

# Device settings

# When running on GPU, assign an entire GPU for each client
# Refer to Flower framework documentation for more details about Flower simulations
# and how to set up the `backend_config`
if device == "cuda":
    backend_config = {
        "client_resources": {
            "num_cpus": 1, 
            "num_gpus": 1
        }
    }
else:
    backend_config = {
        "client_resources": {
            "num_cpus": 1, 
            "num_gpus": 0
        }
    }

print(f"Training on {device}")

partition_name = 'iid' if partition_type == 'iid' else 'non_iid'
model_save_path =  CHECKPOINT_DIR + f"/fl_dino_v4/{partition_name}"
model_config={
    "head_layers": 3,
    "head_hidden_size": 512,
    "dropout_rate": 0.0,
    "unfreeze_blocks": 0,
}


from fl_g13.fl_pytorch.datasets import reset_partition
from fl_g13.modeling import load_or_create
from torch.optim import SGD
from tqdm import tqdm

from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app

reset_partition()

checkpoint_dir = f"{model_save_path}/{Ncs}_{J}_dynamic-quorum"
os.makedirs(checkpoint_dir, exist_ok=True)

client = get_client_app(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        partition_type=partition_type,
        local_epochs=1,
        local_steps=J,
        batch_size=batch_size,
        num_shards_per_partition=num_shards_per_partition,
        scheduler=scheduler,
        verbose=0,
        mask=mask,
        model_editing_batch_size=model_editing_batch_size,
        mask_func=None,
        strategy=strategy,
    )

compute_round = num_rounds + 1 - start_epoch
server = get_server_app(
    global_mask = sum_mask,
    num_total_clients = 100,
    quorum_increment = 5,
    quorum_update_frequency = 1,
    initial_quorum = 1,
    
    checkpoint_dir=checkpoint_dir,
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
    device = device,
    use_wandb=use_wandb,
    wandb_config = wandb_config,
    save_every=save_every,
    prefix = name,
    evaluate_each=evaluate_each,
    model = model,
    start_epoch=start_epoch,
    strategy=strategy
)


from flwr.simulation import run_simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config
)

