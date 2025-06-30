#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import flwr
import torch
import dotenv
import os

from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.fl_pytorch import build_fl_dependencies

from fl_g13.fl_pytorch.editing import load_mask
from fl_g13.modeling import load_or_create

from fl_g13.editing.masking import mask_dict_to_list

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM


dotenv.load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

build_fl_dependencies()


CHECKPOINT_DIR = dotenv.dotenv_values()['CHECKPOINT_DIR']

J = 8
partition_type = 'shard'
shards = 1
strategy = 'sum'
mask_type = 'global'
mask_sparsity = 0.7
mask_rounds = 3

mask_name = f'sum_{shards}_{J}_{mask_type}_{mask_sparsity}_{mask_rounds}.pth'
mask_file_name = CHECKPOINT_DIR + '/masks/' + mask_name

model_save_path = CHECKPOINT_DIR + f"/fl/non-iid/{shards}_{J}"

model, start_epoch = load_or_create(
    path=model_save_path,
    model_class=BaseDino,
    model_config=None,
    optimizer=None,
    scheduler=None,
    device=DEVICE,
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
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-5
)
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=8,
    eta_min=1e-5
)


sum_mask = load_mask(mask_file_name)
sum_mask = mask_dict_to_list(model, sum_mask)


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sparsity_over_quorum_plot(sum_mask, mask_name):
    def compute_sparsity_given_quorum(mask, quorum):
        assert 0 < quorum <= 100
        
        global_mask = [(layer_sum >= quorum).float() for layer_sum in mask]
        total_params = sum(np.prod(layer.shape) for layer in global_mask)
        total_non_zero = sum(layer.cpu().numpy().nonzero()[0].size for layer in global_mask)
        return 1.0 - (total_non_zero / total_params)

    all_sparsity = [compute_sparsity_given_quorum(sum_mask, quorum) for quorum in tqdm(range(1, 101), desc = 'Quormum')]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), all_sparsity, '-')
    plt.xlabel('Quorum')
    plt.ylabel('Sparsity')
    plt.title(mask_name)
    plt.grid(True)
    plt.show()


sparsity_over_quorum_plot(sum_mask, mask_name)

