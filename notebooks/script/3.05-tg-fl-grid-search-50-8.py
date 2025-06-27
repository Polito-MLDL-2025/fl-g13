#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import flwr
import torch
import dotenv
import wandb
from itertools import product

from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM

from fl_g13.fl_pytorch import build_fl_dependencies

from fl_g13.fl_pytorch.datasets import reset_partition
from fl_g13.modeling import load_or_create
from fl_g13.editing import SparseSGDM
from torch.optim import SGD

from fl_g13.fl_pytorch import get_client_app, get_server_app
from flwr.simulation import run_simulation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

dotenv.load_dotenv()
build_fl_dependencies()

if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
else:
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}


# login by key in .env file
WANDB_API_KEY = dotenv.dotenv_values()["WANDB_API_KEY"]
wandb.login(key=WANDB_API_KEY)

# Load checkpoint from .env file
CHECKPOINT_DIR = dotenv.dotenv_values()["CHECKPOINT_DIR"]


# Model config
## Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 12

## Training Hyper-parameters
batch_size = 64
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5

# FL config
K = 100
C = 0.1
J = 8
num_shards_per_partition = 50 # Nc
partition_type = 'shard'

num_rounds = 5

## Server App config
save_every = num_rounds # only save last checkpoint
evaluate_each = 1
fraction_fit = C        # 0.1
fraction_evaluate = C   # 0.1
min_fit_clients = 10
min_evaluate_clients = 5
min_available_clients = 10

# model editing config
model_editing = True
mask_types = ['local', 'global']
sparsities = [0.7, 0.8, 0.9]
calibration_rounds = [1, 3]
model_editing_batch_size = 1
mask = None

## simulation run config
NUM_CLIENTS = 100
MAX_PARALLEL_CLIENTS = 10

## Base model location
# The 200-epoch model folder
# Ensure that the most recent file is the correct one
model_save_path = CHECKPOINT_DIR + f"/fl/non-iid/{num_shards_per_partition}_{J}"


# Run simulations
for m_calibration_rounds, m_type, m_sparsity in product(calibration_rounds, mask_types, sparsities):
    reset_partition()
    
    print('-' * 200)
    print(f" Nc={num_shards_per_partition}, J={J}, mask_type={m_type}, sparsity={m_sparsity}, mask_calibration_round={m_calibration_rounds}\n")
    
    model_checkpoint = CHECKPOINT_DIR + f"/fl/non-iid/GS/{num_shards_per_partition}_{J}_{m_type}_{m_sparsity}_{m_calibration_rounds}"
    
    # Load Base model
    model, start_epoch = load_or_create(
        path=model_save_path,
        model_class=BaseDino,
        model_config=None,
        optimizer=None,
        scheduler=None,
        device=DEVICE,
    )
    model.to(DEVICE)

    if model_editing:
        # Create a dummy mask for SparseSGDM
        dummy_mask = [torch.ones_like(p, device=p.device) for p in model.parameters()]
        
        optimizer = SparseSGDM(
            model.parameters(),
            mask=dummy_mask,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Unfreeze entire model for model_editing
    model.unfreeze_blocks(unfreeze_blocks)

    # Wandb settings
    use_wandb = True
    run_name = f"fl_gs_{num_shards_per_partition}_{J}_{m_type}_{m_sparsity}_{m_calibration_rounds}"
    wandb_config = {
        # wandb param
        'name': run_name,
        'project_name': f"fl_v5_{num_shards_per_partition}_{J}_grid_search",
        'run_id': run_name,
        
        # fl config
        "fraction_fit": fraction_fit,
        "lr": lr,
        "momentum": momentum,
        'weight_decay': weight_decay,
        'partition_type': partition_type,
        'K': K,
        'C': C,
        'J': J,
        'Nc': num_shards_per_partition,
        
        # model config
        'head_layers': head_layers,
        'head_hidden_size': head_hidden_size,
        'dropout_rate': dropout_rate,
        'unfreeze_blocks': unfreeze_blocks,
        
        # model editing config
        'model_editing_batch_size': model_editing_batch_size,
        'mask_calibration_round': m_calibration_rounds,
        'mask_type': m_type,
        'sparsity': m_sparsity
    }

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
        scheduler=None,
        model_editing=model_editing,
        mask_type=m_type,
        sparsity=m_sparsity,
        mask=mask,
        model_editing_batch_size=model_editing_batch_size,
        mask_func=None,
        mask_calibration_round=m_calibration_rounds
    )
    
    server = get_server_app(
        checkpoint_dir=model_checkpoint,
        model_class=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        device=DEVICE,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
        save_every=save_every,
        prefix='grid_search',
        evaluate_each=evaluate_each,
        model= model,
        start_epoch= start_epoch
    )
    
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config
    )

    if use_wandb:
        wandb.finish()

