#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


get_ipython().run_line_magic('pip', 'install -e ..')


import wandb
import dotenv
import os

from pathlib import Path

import torch
import flwr
from flwr.simulation import run_simulation

from fl_g13.architectures import BaseDino
from fl_g13.fl_pytorch.client_app import get_client_app
from fl_g13.fl_pytorch.server_app import get_server_app

from fl_g13.fl_pytorch import build_fl_dependencies

dotenv.load_dotenv() # Load API Key
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
build_fl_dependencies()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}")

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if device == "cuda":
    backend_config["client_resources"] = {"num_cpus": 1, "num_gpus": 1}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# ## Hyperparameters

from fl_g13.modeling import load_or_create
from fl_g13.editing import SparseSGDM
from torch.optim import SGD

# Model Hyper-parameters
model_config={
    "head_layers": 3,
    "head_hidden_size": 512,
    "dropout_rate": 0.0,
    "unfreeze_blocks": 0,
}

# Training Hyper-parameters
batch_size = 64
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8 #! Unused
eta_min = 1e-5 #! Unused

# Model editing Hyper-parameters
model_editing = True
mask_types = ['global', 'local']
sparsities = [0.7, 0.8, 0.9]
calibration_rounds = [1, 3]
mask = None
model_editing_batch_size = 16

# Federated Hyper-parameters
K = 100
C = 0.1
J = 8
Nc = 50

save_every = 5
fraction_fit = C  # Sample of available clients for training
fraction_evaluate = 0.1  # Sample 50% of available clients for evaluation
min_fit_clients = 10  # Never sample less than 10 clients for training
min_evaluate_clients = 5  # Never sample less than 5 clients for evaluation
min_available_clients = 10  # Wait until all 10 clients are available

num_rounds = 100 + 40 # Add the current number of epochs
evaluate_each = 5
partition_type = 'shard'
NUM_CLIENTS = K

# Wandb config
use_wandb = True
project_name = "FL_Dino_CIFAR100_masking_grid_search_v4_non_iid_50_8"

current_path = Path.cwd()
model_save_path = current_path / f"../models/fl_dino_masking_grid_search_v4/non_iid"


# ## Training

for mask_type in mask_types:
    for sparsity in sparsities:
        for mask_calibration_round in calibration_rounds:
            print('-' * 200)
            print(f"Training Non IId model")
            print(f"Nc: {Nc}, J: {J}")
            checkpoint_dir = f"{model_save_path}/{Nc}_{J}_{mask_type}_{mask_calibration_round}_{sparsity}"
            print(f'checkpoint_dir:{checkpoint_dir}')
            
            # Model
            model, start_epoch = load_or_create(
                path='/home/massimiliano/Projects/fl-g13/models/fl_dino_v4/non_iid/50_8/editing/fl_fl_baseline_BaseDino_epoch_100.pth', #checkpoint_dir,
                model_class=BaseDino,
                model_config=model_config,
                optimizer=None,
                scheduler=None,
                device=device,
                verbose=True,
            )
            model.to(device)

            optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
            criterion = torch.nn.CrossEntropyLoss()

            ## Unfreeze blocks
            num_blocks = 12
            model.unfreeze_blocks(num_blocks)
            num_shards_per_partition = Nc

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params}")

            os.makedirs(checkpoint_dir, exist_ok=True)
            name = f"FL_Dino_Baseline_model_non_iid_{Nc}_{J}_{mask_type}_{mask_calibration_round}_{sparsity}"
            
            wandb_config = {
                # Wandb Params
                'name': name,
                'project_name': project_name,
                'run_id': f"{name}",
                # Federated Learning param
                "fraction_fit": fraction_fit,
                'partition_type': partition_type,
                'K': K,
                'C': C,
                'J': J,
                'Nc': Nc,
                # Model editing params
                'model_editing': model_editing,
                'mask_type': mask_type,
                'sparsity': sparsity,
                'model_editing_batch_size': model_editing_batch_size,
                'calibration_rounds': mask_calibration_round,
                # Training params
                'lr': lr,
                'momentum': momentum,
            }

            if model_editing:
                # Create a dummy mask for SparseSGDM
                init_mask = [torch.ones_like(p, device=p.device) for p in
                            model.parameters()]  # Must be done AFTER the model is moved to the device
                # Optimizer
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
                device=device,
                partition_type=partition_type,
                local_epochs=1,
                local_steps=J,
                batch_size=batch_size,
                num_shards_per_partition=num_shards_per_partition,
                scheduler=None, #! Clients wont use scheduler, as it doesnt make sense here
                verbose=0,
                model_editing=model_editing,
                mask_type=mask_type,
                sparsity=sparsity,
                mask=mask,
                mask_calibration_round=mask_calibration_round,
                model_editing_batch_size=model_editing_batch_size,
                mask_func=None
            )

            compute_round = num_rounds + 1 - start_epoch
            server = get_server_app(
                checkpoint_dir=checkpoint_dir,
                model_class=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=None, #! Server wont use scheduler, as it doesnt make sense here
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
            
            wandb.finish()




