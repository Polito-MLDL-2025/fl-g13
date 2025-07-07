#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Import cell

import flwr
import torch
import dotenv
import wandb

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


# # Configurations

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


# # Training parameters definition

# FL config
K = 100
C = 0.1
Js = [4, 8, 16]
partition_type = 'iid'

num_rounds = 1

## Server App config
save_every = 5
evaluate_each = 5
fraction_fit = C        # 0.1
fraction_evaluate = C   # 0.1
min_fit_clients = 10
min_evaluate_clients = 5
min_available_clients = 10

# model editing config
model_editing = False
mask = None

## simulation run config
NUM_CLIENTS = 100
MAX_PARALLEL_CLIENTS = 10


# # Load model

# Model config
## Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 0

## Training Hyper-parameters
batch_size = 64
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5

def load_model_for_simulation(J):
    ## Base model location
    model_save_path = CHECKPOINT_DIR + f"/fl/iid/warmup/iid_{J}"
    
    model, start_epoch = load_or_create(
        path=model_save_path,
        model_class=BaseDino,
        model_config=None,
        optimizer=None,
        scheduler=None,
        device=DEVICE,
        verbose=True,
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
    scheduler = None
    
    return model, start_epoch, optimizer, criterion, scheduler


# # Run Flower Simulation

# Run all the simulations, one after the other
for J in Js:
    print(f'||| Warm-up for iid_{J} |||\n')
    
    reset_partition()
    
    model, start_epoch, optimizer, criterion, scheduler = load_model_for_simulation(J)
    model_checkpoint = CHECKPOINT_DIR + f"/fl/iid/warmup/{J}"

    # Wandb settings
    use_wandb = False
    run_name = f"fl_iid_{J}"
    wandb_config = {
        # wandb param
        'name': run_name,
        'project_name': f"fl_v5_iid_warmup",
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
        
        # model config
        'head_layers': head_layers,
        'head_hidden_size': head_hidden_size,
        'dropout_rate': dropout_rate,
        'unfreeze_blocks': unfreeze_blocks,
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
        scheduler=scheduler,
        model_editing=model_editing,
        mask=mask,
        mask_func=None
    )

    server = get_server_app(
        checkpoint_dir=model_checkpoint,
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
        device=DEVICE,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
        save_every=save_every,
        prefix='warmup',
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

