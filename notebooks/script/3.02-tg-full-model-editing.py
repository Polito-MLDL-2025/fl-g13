#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fl_g13.config import RAW_DATA_DIR

from fl_g13.modeling import train, load, eval, get_preprocessing_pipeline

from fl_g13.architectures import BaseDino

from fl_g13.editing import SparseSGDM
from fl_g13.editing import create_mask, mask_dict_to_list

import dotenv


dotenv.load_dotenv()

train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# # Define model to edit

# Move to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Settings
name="arcanine"
CHECKPOINT_DIR = dotenv.dotenv_values()["CHECKPOINT_DIR"]
start_epoch = 1
num_epochs = 30
save_every = 1
backup_every = None

# Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 1

# Training Hyper-parameters
batch_size = 128
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = BaseDino(
    head_layers=head_layers, 
    head_hidden_size=head_hidden_size, 
    dropout_rate=dropout_rate, 
    unfreeze_blocks=unfreeze_blocks
    )
model.to(device)

# Optimizer, scheduler, and loss function
mask = [torch.ones_like(p, device=p.device) for p in model.parameters()] # Must be done AFTER the model is moved to CUDA
optimizer = SparseSGDM(
    model.parameters(),
    mask=mask,
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay
    )
scheduler = CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=T_max, 
    eta_min=eta_min
    )
criterion = CrossEntropyLoss()

loading_epoch = 8
loading_model_path =  f"{CHECKPOINT_DIR}/{name}/{name}_BaseDino_epoch_{loading_epoch}.pth"
model, start_epoch = load(
    loading_model_path,
    model_class=BaseDino,
    device=device,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True
)
model.to(device)

model_metrics_path=f"{CHECKPOINT_DIR}/{name}/{name}_BaseDino_epoch_{loading_epoch}.loss_acc.json"


# # Define model editing

# ## Create mask

import os

def get_centralized_model_mask(model, dataloader, sparsity, mask_type, calibration_rounds, file_path = 'centralized_model_mask.pth', verbose = False):
    if file_path and os.path.isfile(file_path):
        if verbose:
            print(f'[CMM] Found {file_path}. Loading mask from memory')
            
        return torch.load(file_path)
    
    # else    
    if verbose:
        print('[CMM] Computing mask')
    mask = create_mask(
        dataloader, 
        model, 
        sparsity = sparsity, 
        mask_type = mask_type, 
        rounds = calibration_rounds, 
        verbose = verbose
    )
    
    if verbose:
        print(f'[CMM] Saving the mask at "{file_path}"')
    torch.save(mask, file_path)
    return mask


sparsity = .7
mask_type = 'local'
calibration_rounds = 3
fisher_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle=True)

me_model_name = f'{name}_{loading_epoch}_{mask_type}_{sparsity}_{calibration_rounds}'
file_path = CHECKPOINT_DIR + f'/masks/{me_model_name}.pth'

# Unfreeze the model before computing the mask
unfreeze_blocks = 12
for param in model.backbone.blocks[-unfreeze_blocks:].parameters():
    param.requires_grad = True
mask = get_centralized_model_mask(model, fisher_dataloader, sparsity, mask_type, calibration_rounds, file_path, verbose = True)
mask_list = mask_dict_to_list(model, mask)


# # Fine-tune the model

def fine_tune(
    starting_model_path, 
    model_name, 
    train_dataloader, 
    test_dataloader, 
    val_dataloader, 
    mask, 
    optimizer, 
    scheduler, 
    criterion, 
    epochs = 10, 
    verbose = 1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    new_model, start_epoch = load(
        path = starting_model_path,
        model_class = BaseDino,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device
    )
    new_model.to(device) # manually move the model to the device

    # unfreeze the model
    for param in new_model.backbone.blocks[-unfreeze_blocks:].parameters():
        param.requires_grad = True

    # Create a new SparseSGDM optimizer
    new_optimizer = SparseSGDM(
        new_model.parameters(), 
        mask = mask, 
        lr = lr,
        momentum = momentum,
        weight_decay = weight_decay
    )

    try: 
        _, _, _, _ = train(
            checkpoint_dir = f'{CHECKPOINT_DIR}/{model_name}',
            name = model_name,
            start_epoch = start_epoch,
            num_epochs = epochs,
            save_every = 1,
            backup_every = None,
            train_dataloader = train_dataloader,
            val_dataloader = val_dataloader,
            model = new_model,
            criterion = criterion,
            optimizer = new_optimizer,
            scheduler = scheduler,
            verbose = verbose,
            with_model_dir = False
        )
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    except Exception as e:
        print(f"Training stopped due to error: {e}")

    # Final eval
    test_loss, test_accuracy, _ = eval(dataloader=test_dataloader, model=new_model, criterion=criterion)

    return test_loss, test_accuracy


# Model editing
me_test_loss, me_test_acc = fine_tune(
    starting_model_path = loading_model_path,
    model_name = me_model_name,
    train_dataloader = train_dataloader,
    test_dataloader = test_dataloader,
    val_dataloader = val_dataloader,
    mask = mask_list,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    epochs = num_epochs - loading_epoch, # to get to 30
    verbose = 1
)

