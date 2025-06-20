#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import flwr
import torch
import dotenv
import wandb
import os

from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.fl_pytorch import build_fl_dependencies

from fl_g13.fl_pytorch.editing import load_or_create_centralized_mask
from fl_g13.editing.masking import mask_dict_to_list

from fl_g13.fl_pytorch.datasets import reset_partition
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app
from flwr.simulation import run_simulation

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM


dotenv.load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

build_fl_dependencies()

# login by key in .env file
WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
wandb.login(key=WANDB_API_KEY)


# # FL CONFIG

CHECKPOINT_DIR = dotenv.dotenv_values()["CHECKPOINT_DIR"]

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
num_shards_per_partition = 50

## Server App config
save_every = 5
fraction_fit = C  # Sample of available clients for training
fraction_evaluate = 0.1  # Sample 50% of available clients for evaluation
min_fit_clients = 10  # Never sample less than 10 clients for training
min_evaluate_clients = 5  # Never sample less than 5 clients for evaluation
min_available_clients = 10  # Wait until all 10 clients are available
device = DEVICE
client_batch_size = 16

# Adaptive Quorum strategy
strategy = 'quorum'
mask_strategy = 'sum'
quorum_increment = 7
quorum_update_frequency = 5
initial_quorum = 1
adaptive_quorum = True
initial_target_sparsity = 0.65
drift_threshold = 0.0005
quorum_patience = 3
force_quorum_update = 15

# model editing config
model_editing = True
mask_type = 'local'
sparsity = 0.7
num_calibration_rounds = 3
mask = None

## simulation run config
NUM_CLIENTS = 100
MAX_PARALLEL_CLIENTS = 10

total_rounds = 300
evaluate_each = 5

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
model_save_path = CHECKPOINT_DIR + f"/fl_dino_v4/{partition_name}/{num_shards_per_partition}_{J}"
model_config={
    "head_layers": 3,
    "head_hidden_size": 512,
    "dropout_rate": 0.0,
    "unfreeze_blocks": 0,
}


from fl_g13.modeling import load_or_create

# Model
model, start_epoch = load_or_create(
    path=model_save_path,
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


file_name = CHECKPOINT_DIR + '/masks/' + f'{num_shards_per_partition}_{J}_{mask_strategy}_mask_{mask_type}_{sparsity}_{num_calibration_rounds}.pth'

sum_mask, _ = load_or_create_centralized_mask(
    model = model,
    strategy = mask_strategy,
    aggregation_fn = None,
    client_partition_type = partition_type,
    client_num_shards_per_partition = num_shards_per_partition,
    client_local_steps = J,
    
    sparsity = sparsity,
    mask_type = mask_type,
    mask_rounds = num_calibration_rounds,
    
    file_name = file_name,
    verbose = True
)
sum_mask = mask_dict_to_list(model, sum_mask)


# # Simulation

reset_partition()

checkpoint_dir = CHECKPOINT_DIR + f"/fl_dino_v4/{partition_name}/dq{'_A' if adaptive_quorum else '_L'}_{num_shards_per_partition}_{J}_{strategy}_{mask_type}_{sparsity}_{num_calibration_rounds}"
os.makedirs(checkpoint_dir, exist_ok=True)

## Wandb config
use_wandb = True
name = f"DQ_{'A' if adaptive_quorum else 'L'}_{num_shards_per_partition}_{J}_{mask_type}_{sparsity}_{num_calibration_rounds}"
print(name)

wandb_config = {
    # wandb param
    'name': name,
    'project_name': "Dynamic-Quorum",
    
    # model config
    "lr": lr,
    "momentum": momentum,
    
    # FL config
    'K': K,
    'C': C,
    'J': J,
    'partition_type': partition_type,
    "fraction_fit": fraction_fit,
    
    # model editing config
    'mask_type': mask_type,
    'sparsity': sparsity,
    'num_calibration_rounds': num_calibration_rounds,
    
    # strategy config
    'adaptive_quorum': adaptive_quorum,
    'initial_quorum': initial_quorum,
    'quorum_update_frequency': quorum_update_frequency,
    'quorum_increment': quorum_increment,
    'drift_threshold': drift_threshold,
    'quorum_patience': quorum_patience
}

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
        model_editing_batch_size=client_batch_size,
        mask_func=None,
        strategy=strategy,
    )

compute_round = total_rounds + 1 - start_epoch

server = get_server_app(
    global_mask = sum_mask,
    num_total_clients = NUM_CLIENTS,
    adaptive_quorum = adaptive_quorum,
    initial_target_sparsity = initial_target_sparsity,
    quorum_increment = quorum_increment,
    quorum_update_frequency = quorum_update_frequency,
    initial_quorum = initial_quorum,
    drift_threshold = drift_threshold,
    quorum_patience = quorum_patience,
    force_quorum_update = force_quorum_update,
    
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


try:
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config
    )
except:
    pass

wandb.finish()

