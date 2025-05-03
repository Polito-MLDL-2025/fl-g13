#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import numpy as np

from fl_g13.config import RAW_DATA_DIR, PROJ_ROOT

from fl_g13.modeling import train, load, eval, plot_metrics, get_preprocessing_pipeline

from fl_g13.architectures import BaseDino

from fl_g13.editing import SparseSGDM
from fl_g13.editing import per_class_accuracy, get_worst_classes, build_per_class_dataloaders
from fl_g13.editing import fisher_scores
from fl_g13.editing import create_gradiend_mask, mask_dict_to_list


train_dataset, val_dataset, test_dataset = get_preprocessing_pipeline(RAW_DATA_DIR)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# # Define the model to edit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_DIR = str(PROJ_ROOT / 'checkpoints')
model_name = 'yamask'
model_checkpoint_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.pth'
model_metrics_path = f'{CHECKPOINT_DIR}/Editing/{model_name}.loss_acc.json'

# Empty model
# Will be replaced with the already trained model from the checkpoint
model = BaseDino(head_layers=5, head_hidden_size=512, dropout_rate=0.0, unfreeze_blocks=1)
model.to(device)

# Hyper-parameters
BATCH_SIZE = 128
LR = 1e-3

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Create a dummy mask for SparseSGDM
mask = [torch.ones_like(p, device = p.device) for p in model.parameters()] # Must be done AFTER the model is moved to the device
# Optimizer, scheduler, and loss function
optimizer = SparseSGDM(model.parameters(), mask = mask, lr = LR)
scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = 20, eta_min = 1e-5)
criterion = CrossEntropyLoss()

# Load the model
model, _ = load(
    path = model_checkpoint_path,
    model_class = BaseDino,
    optimizer = optimizer,
    scheduler = scheduler,
    device = device
)
model.to(device) # manually move the model to the device
print(f'\nModel {model_name} loaded from checkpoint.')


# Compute test accuracy
test_loss, test_accuracy, _ = eval(test_dataloader, model, criterion)

print(f'Test loss: {test_loss:.3f}')
print(f'Test accuracy: {100*test_accuracy:.2f}%')

# Plot training results
plot_metrics(path = model_metrics_path)


# # Model editing

# ## Compute per-class accuracy
# Find the class in which the model is underperforming

class_acc = per_class_accuracy(test_dataloader, model)
print(f'\nClass accuracy (first 10 classes): {class_acc[:10]}') # Output preview


N_worst = 3 # How many classes to fine-tune
worst_classes = get_worst_classes(class_acc, N_worst)
print(f"Worst classes: {worst_classes}")


# Note that the batch size in this case is 32 by default
# Since the dataloaders are specific to the classes, a smaller batch size is better
classes_dataloaders = build_per_class_dataloaders(train_dataset, worst_classes)


# ## Compute Fisher Sentitivity (per-class)

def compute_score_per_classes(model, classes, classes_dataloaders):
    score_per_class = {}

    for cls in classes:
        print(f"Computing scores for class {cls}")
        scores = fisher_scores(classes_dataloaders[cls], model)
        score_per_class[cls] = scores

    return score_per_class

scores_per_class = compute_score_per_classes(model, worst_classes, classes_dataloaders)


# ## Create Gradient Masks

def compute_masks_per_classes(classes, scores_per_class):
    global_masks, local_masks = {}, {}

    for cls in classes:
        # print(f"Computing Mask for class {cls}")
        global_masks[cls] = create_gradiend_mask(scores_per_class[cls], mask_type = 'global')
        local_masks[cls] = create_gradiend_mask(scores_per_class[cls], mask_type = 'local')

    return global_masks, local_masks

global_masks, local_masks = compute_masks_per_classes(worst_classes, scores_per_class)


# Convert the masks to a list, as required by SparseSGDM
def convert_masks_to_list(classes, masks_per_class):
    masks_lists = {}

    for cls in classes:
        # print(f"Computing Mask for class {cls}")
        mask = mask_dict_to_list(model, masks_per_class[cls])
        masks_lists[cls] = mask

    return masks_lists

global_masks_list = convert_masks_to_list(worst_classes, global_masks)
local_masks_list = convert_masks_to_list(worst_classes, local_masks)


# ## Fine tune the model on the choosen classes

import json

def fine_tuned_model(class_to_fine_tune, train_dataloader, mask, optimizer, scheduler, criterion, epochs = 10, verbose = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    new_model, _ = load(
        path = model_checkpoint_path,
        model_class = BaseDino,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device
    )
    new_model.to(device) # manually move the model to the device

    # Create a new SparseSGDM optimizer
    new_optimizer = SparseSGDM(new_model.parameters(), mask = mask, lr = LR)

    _, _, _, _ = train(
        checkpoint_dir = CHECKPOINT_DIR,
        name = f'{model_name}_{class_to_fine_tune}',
        start_epoch = 1,
        num_epochs = epochs,
        save_every = epochs,
        backup_every = None,
        train_dataloader = train_dataloader,
        val_dataloader = None,
        model = new_model,
        criterion = criterion,
        optimizer = new_optimizer,
        scheduler = None, # No scheduler needed, too few epochs
        verbose = verbose
    )

    # Compute per-class accuracy
    class_acc = per_class_accuracy(test_dataloader, new_model)

    return class_acc

def fine_tune(classes, classes_dataloaders, masks, optimizer, scheduler, criterion, mask_type):
    # Fine-tune the model on the worst classes
    for cls in classes:
        print(f"Fine-tuning model on class {cls}")

        # Get the dataloaders for the current class
        train_dataloader = classes_dataloaders[cls]

        # Get the mask for the current class
        mask = masks[cls]

        # Fine-tune the model
        new_class_acc = fine_tuned_model(
            class_to_fine_tune = cls,
            train_dataloader = train_dataloader,
            mask = mask,
            optimizer = optimizer,
            scheduler = scheduler,
            criterion = criterion
        )

        # Compare results with the original model
        new_test_accuracy = np.mean(new_class_acc)

        print(f'\nTest accuracy: {100*new_test_accuracy:.2f}% (original: {100*test_accuracy:.2f}%)')
        # Print print accuracy for the specific class
        print(f'Accuracy for class {cls}: {100*new_class_acc[cls]:.2f}% (original: {100*class_acc[cls]:.2f}%)')
        # Print other classes accuracy if the new model is worse than the original
        count = sum([1 for i in range(len(new_class_acc)) if new_class_acc[i] < class_acc[i] and i != cls])
        print(f'New model is worse in {count} classes, wrt the original model')

        # Save to file the per-class accuracy difference
        # Create a dictionary with new_class_acc, class_acc, and class_idx
        accuracy_data = {
            "class_idx": list(range(100)),
            "new_class_acc": list(new_class_acc),
            "class_acc": list(class_acc)
        }
        output_file = f"{CHECKPOINT_DIR}/Editing/{model_name}/accuracy_comparison_{cls}_{mask_type}.json"

        # Save the dictionary to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(accuracy_data, json_file, indent=4)
        print(f"Accuracy data saved to {output_file}\n\n")

print('Fine-tune with global masks')
fine_tune(worst_classes, classes_dataloaders, global_masks_list, optimizer, scheduler, criterion, 'global')
print('\n\nFine-tune with local masks')
fine_tune(worst_classes, classes_dataloaders, local_masks_list, optimizer, scheduler, criterion, 'local')

