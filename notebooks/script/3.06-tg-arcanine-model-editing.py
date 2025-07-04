#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Import cell

import os
import torch
import dotenv

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fl_g13.config import RAW_DATA_DIR

from fl_g13.modeling import train, load, eval, get_preprocessing_pipeline, plot_metrics

from fl_g13.architectures import BaseDino

from fl_g13.editing import SparseSGDM
from fl_g13.editing import create_mask, mask_dict_to_list


# # Configurations

dotenv.load_dotenv()
CHECKPOINT_DIR = dotenv.dotenv_values()["CHECKPOINT_DIR"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# # Validation datasets

train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# # Full training datasets

full_train_dataset, full_test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR, do_full_training = True)

print(f"Full Train dataset size: {len(full_train_dataset)}")
print(f"Full Test dataset size: {len(full_test_dataset)}")


# # Define model to edit

# Settings
name = "arcanine"

# Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 0

# Training Hyper-parameters
batch_size = 64
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5

# Base Model
model = BaseDino(
    head_layers=head_layers, 
    head_hidden_size=head_hidden_size, 
    dropout_rate=dropout_rate, 
    unfreeze_blocks=unfreeze_blocks
)
model.to(DEVICE)

# Optimizer, scheduler, and loss function
dummy_mask = [torch.ones_like(p, device=p.device) for p in model.parameters()]
optimizer = SparseSGDM(
    model.parameters(),
    mask=dummy_mask,
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

# Load arcanine
loading_epoch = 10
loading_model_path =  f"{CHECKPOINT_DIR}/{name}/{name}_BaseDino_epoch_{loading_epoch}.pth"
model, start_epoch = load(
    loading_model_path,
    model_class=BaseDino,
    device=DEVICE,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True
)
model.to(DEVICE)


# # Create mask

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


sparsity = .9
mask_type = 'global'
calibration_rounds = 3
unfreeze_blocks = 12
fisher_dataloader = DataLoader(full_train_dataset, batch_size = 1, shuffle=True)

me_model_name = f'{name}_{loading_epoch}_{mask_type}_{sparsity}_{calibration_rounds}'
file_path = CHECKPOINT_DIR + f'/masks/{me_model_name}.pth'

# Unfreeze the model before computing the mask
model.unfreeze_blocks(unfreeze_blocks)
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
    unfreeze_blocks = 12
    new_model.unfreeze_blocks(unfreeze_blocks)

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
    if test_dataloader:
        test_loss, test_accuracy, _ = eval(dataloader=test_dataloader, model=new_model, criterion=criterion)
        return test_loss, test_accuracy
    else:
        return -1, -1


batch_size = 64

# Validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Full Training dataloaders
full_train = DataLoader(full_train_dataset, batch_size = batch_size, shuffle = True)
full_test = DataLoader(full_test_dataset, batch_size = batch_size, shuffle = False)


# # Fine tuning

# ## Validation

val_epoch = 30
val_model_name = 'val_arcanine'
starting_model_path = f"{CHECKPOINT_DIR}/{val_model_name}/{val_model_name}_BaseDino_epoch_{loading_epoch}.pth"

# Validation
_, _ = fine_tune(
    starting_model_path = starting_model_path,
    model_name = f'me_{me_model_name}',
    train_dataloader = train_dataloader,
    test_dataloader = None,
    val_dataloader = val_dataloader,
    mask = mask_list,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    epochs = val_epoch - loading_epoch, # to get to 30
    verbose = 1
)

# plot metrics
metrics_data = f"{CHECKPOINT_DIR}/{f'me_{me_model_name}'}/{f'me_{me_model_name}'}_BaseDino_epoch_{val_epoch}.loss_acc.json"
plot_metrics(path = metrics_data)


# # Full training

num_epochs = 30
base_model = 'arcanine'
starting_model_path = f"{CHECKPOINT_DIR}/{base_model}/{base_model}_BaseDino_epoch_{loading_epoch}.pth"
me_model_name = 'arcanine_talos'
me_test_loss, me_test_acc = fine_tune(
    starting_model_path = loading_model_path,
    model_name = me_model_name,
    train_dataloader = full_train,
    test_dataloader = full_test,
    val_dataloader = None,
    mask = mask_list,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion = criterion,
    epochs = num_epochs - loading_epoch, # to get to 30
    verbose = 1
)

print(
    f"🔍 Test Results:\n"
    f"\t📉 Test Loss: {me_test_loss:.4f}\n"
    f"\t🎯 Test Accuracy: {100 * me_test_acc:.2f}%"
)

# Plot metrics
metrics_data = f'{CHECKPOINT_DIR}/{me_model_name}/{me_model_name}_BaseDino_epoch_{num_epochs}.loss_acc.json'
plot_metrics(path = metrics_data)

