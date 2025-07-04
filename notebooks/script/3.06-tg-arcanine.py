#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Import cell

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fl_g13.config import RAW_DATA_DIR
from fl_g13.modeling import train, eval, get_preprocessing_pipeline, plot_metrics

from fl_g13.architectures import BaseDino
from fl_g13.editing import SparseSGDM

import dotenv


# # Configurations

dotenv.load_dotenv()
CHECKPOINT_DIR = dotenv.dotenv_values()["CHECKPOINT_DIR"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# # Validation phase

train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# Settings
name="val_arcanine"
start_epoch = 1
num_epochs = 25
save_every = 1
backup_every = None

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

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# Model
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


try:
    _, _, _, _ = train(
        checkpoint_dir=f'{CHECKPOINT_DIR}/{name}',
        name=name,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        save_every=save_every,
        backup_every=backup_every,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=1,
        with_model_dir = False
    )
except KeyboardInterrupt:
    print("Training interrupted manually.")
except Exception as e:
    print(f"Training stopped due to error: {e}")


metrics_data = f'{CHECKPOINT_DIR}/{name}/{name}_BaseDino_epoch_{num_epochs}.loss_acc.json'
plot_metrics(path = metrics_data)


# # Full training

full_train_dataset, full_test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR, do_full_training=True)

print(f"Train dataset size: {len(full_train_dataset)}")
print(f"Test dataset size: {len(full_test_dataset)}")


# Settings
name = "arcanine"
start_epoch = 1
num_epochs = 10 # From the validation training
save_every = 1
backup_every = None

# Model Hyper-parameters
head_layers = 3
head_hidden_size = 512
dropout_rate = 0.0
unfreeze_blocks = 0

# Training Hyper-parameters
batch_size = 128
lr = 1e-3
momentum = 0.9
weight_decay = 1e-5
T_max = 8
eta_min = 1e-5

# Dataloaders
train_dataloader = DataLoader(full_train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(full_test_dataset, batch_size = batch_size, shuffle = False)

# Model
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


try:
    _, _, _, _ = train(
        checkpoint_dir=f'{CHECKPOINT_DIR}/{name}',
        name=name,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        save_every=save_every,
        backup_every=backup_every,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=1,
        with_model_dir = False
    )
except KeyboardInterrupt:
    print("Training interrupted manually.")
except Exception as e:
    print(f"Training stopped due to error: {e}")


# Final test
test_loss, test_accuracy, _ = eval(test_dataloader, model, criterion, verbose = 1)
print(
    f"🔍 Test Results:\n"
    f"\t📉 Test Loss: {test_loss:.4f}\n"
    f"\t🎯 Test Accuracy: {100 * test_accuracy:.2f}%"
)

metrics_data = f'{CHECKPOINT_DIR}/{name}/{name}_BaseDino_epoch_{num_epochs}.loss_acc.json'
plot_metrics(path = metrics_data)

