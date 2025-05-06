#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from fl_g13.editing import SparseSGDM
from fl_g13.editing import create_gradiend_mask
from fl_g13.editing import fisher_scores
from fl_g13.modeling import eval


from torchvision import transforms
from fl_g13.fl_pytorch.datasets import load_datasets

partition_id = 1
num_partitions = 100
partition_type = 'iid'
batch_size = 128
num_shards_per_partition = 6
train_test_split_ratio = 0.2


def get_transforms():
    """Return a function that apply standard transformations to images."""

    def apply_transforms(batch):
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        batch["fine_label"] = [int(lbl) for lbl in batch["fine_label"]]

        return batch

    return apply_transforms


trainloader, valloader = load_datasets(
    partition_id,
    num_partitions,
    partition_type=partition_type,
    batch_size=batch_size,
    num_shards_per_partition=num_shards_per_partition,
    train_test_split_ratio=train_test_split_ratio,
    transform=get_transforms
)


len(trainloader)


from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_path = Path.cwd()
model_test_path = current_path / "../models/model_test"
model_test_path.resolve()


from fl_g13.fl_pytorch.model import TinyCNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import SGD
from fl_g13.modeling import load_or_create

BATCH_SIZE = 128
LR = 1e-3
checkpoint_dir = model_test_path.resolve()
model_class = TinyCNN
# Optimizer, scheduler, and loss function
model = TinyCNN()
optimizer = SGD(model.parameters(), lr=LR)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=8,  # First restart after 8 epochs
    T_mult=2,  # Double the interval between restarts each time
    eta_min=1e-5  # Minimum learning rate after annealing
)
criterion = CrossEntropyLoss()
# Load the model
model, start_epoch = load_or_create(
    path=checkpoint_dir,
    model_class=model_class,
    device=device,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
)
model.to(device)

# Create a dummy mask for SparseSGDM
mask = [torch.ones_like(p, device=p.device) for p in
        model.parameters()]  # Must be done AFTER the model is moved to the device
# Optimizer, scheduler, and loss function
optimizer = SparseSGDM(
    model.parameters(),
    mask=mask,
    lr=LR,
    momentum=0.9,
    weight_decay=1e-5
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=8,
    eta_min=1e-5
)
criterion = CrossEntropyLoss()



for batch_idx, (X, y) in enumerate(valloader):
    X, y = X.to(device), y.to(device)


## compute fisher scores

scores = fisher_scores(dataloader=valloader, model=model, loss_fn=criterion,verbose=1)
mask = create_gradiend_mask(class_score=scores, sparsity=0.2, mask_type='global')


mask


## eval model before run editing
eval(dataloader=valloader, model=model, criterion=criterion)


from fl_g13.editing import mask_dict_to_list

mask_list = mask_dict_to_list(model, mask)

optimizer = SparseSGDM(
    model.parameters(),
    mask=mask_list,
    lr=LR,
    momentum=0.9,
    weight_decay=1e-5
)


mask_list


from fl_g13.modeling import train
checkpoint_dir_edit = f'{checkpoint_dir}/edit'
name = "model_editing"
epochs= 10
_, _, _, _ = train(
        checkpoint_dir = checkpoint_dir_edit,
        name = name,
        start_epoch = 1,
        num_epochs = epochs,
        save_every = epochs,
        backup_every = None,
        train_dataloader = trainloader,
        val_dataloader = None,
        model = model,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = None, # No scheduler needed, too few epochs
        verbose = 1
    )


eval(dataloader=valloader, model=model, criterion=criterion)


mask_list


from fl_g13.editing.masking import compress_mask_sparse

compressed = compress_mask_sparse(mask_list)
compressed


from fl_g13.editing.masking import uncompress_mask_sparse
uncompress_mask_sparse(compressed,device=device)




