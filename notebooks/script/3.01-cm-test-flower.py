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
from fl_g13.fl_pytorch import get_client_app, get_server_app
from fl_g13.fl_pytorch import build_fl_dependencies

print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

build_fl_dependencies() #! Remind to always put this, it will download Dino dependencies for client


# Settings
CHECKPOINT_DIR = "/home/massimiliano/Projects/fl-g13/checkpoints"

# Model hyper-parameters
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
batch_size = 64 # Batch size for training #! Let's stick to 64 to make training fit also on RTX 3070
local_epochs = 2 # Number of local epochs per client
number_of_rounds = 5 # Total number of federated learning rounds
fraction_fit = 1 # Fraction of clients participating in training per round
fraction_evaluate = 0.1 # Fraction of clients participating in evaluation per round
number_of_clients = 3 # Total number of clients in the simulation
min_num_clients = 2 # Minimum number of clients required for training and evaluation
partition_type = "iid" # Partitioning strategy for the dataset (e.g., "iid" or "shard")
num_shards_per_partition = 6 # Number of shards per partition (used when partition_type is "shard")
use_wandb = False # Whether to use Weights & Biases (wandb) for experiment tracking (#!TODO, double check it works)

# Device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
backend_config = {
    "client_resources": {
        "num_cpus": 1, 
        "num_gpus": 0
    }
}

# When running on GPU, assign an entire GPU for each client
# Refer to Flower framework documentation for more details about Flower simulations
# and how to set up the `backend_config`
if device == "cuda":
    backend_config["client_resources"] = {"num_cpus": 1, "num_gpus": 1}

print(f"Training on {device}")


# Model
model = BaseDino(
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
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=scheduler,
    device=device, 
    partition_type=partition_type, 
    batch_size=batch_size,
    num_shards_per_partition=num_shards_per_partition,
    local_epochs=local_epochs,
    model_editing=False,
)
server_app = get_server_app(
    checkpoint_dir=CHECKPOINT_DIR,
    prefix='aron', #! Introduced, you are force to pass this to avoid overwrites, if you pass an already used name it will load the most recent checkpoint
    model_class=model.__class__,
    model_config=model.get_config(), 
    optimizer=optimizer, 
    criterion=criterion, 
    scheduler=scheduler,
    device=device, 
    save_every=1,
    save_with_model_dir=False, #! Introduced: will save under {checkpoint path provided}/BaseDino/ dir if set to True
    num_rounds=number_of_rounds, 
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    min_fit_clients=min_num_clients,
    min_evaluate_clients=min_num_clients,
    min_available_clients=number_of_clients,
    use_wandb=False,
    wandb_config=None,
)


run_simulation(
    client_app=client_app,
    server_app=server_app,
    num_supernodes=number_of_clients,
    backend_config=backend_config,
)




