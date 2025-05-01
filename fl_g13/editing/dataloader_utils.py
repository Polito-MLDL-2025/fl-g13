import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import Module

from sklearn.metrics import confusion_matrix

import numpy as np

from tqdm import tqdm


def per_class_accuracy(
    dataloader: DataLoader,     # DataLoader providing the evaluation dataset
    model: Module,              # The model to evaluate
    verbose: int = 1            # If > 0, display progress
):
    # Get the device where the model is located
    device = next(model.parameters()).device
    # Set the model to evaluation mode
    model.eval()
    
    # Set the for loop iterator according to the verbose flag
    if verbose == 1:
        # Default, use tqdm with progress bar
        batch_iterator = tqdm(dataloader, desc = 'Per Class Accuracy', unit = 'batch')
    else:
        # No progress bar
        batch_iterator = dataloader

    # Initialize variables
    all_preds, all_labels = [], []
    total_batches = len(dataloader)
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(batch_iterator):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
            # Verbose == 2 print progress every 10 batches
            if verbose == 2 and (batch_idx + 1) % 10 == 0:
                print(f"  ↳ Batch {batch_idx + 1}/{total_batches} | Predictions collected")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    conf = confusion_matrix(all_labels, all_preds)
    class_acc = conf.diagonal() / conf.sum(axis=1)
    return class_acc

def get_worst_classes(class_accuracy, num_classes):
    return np.argsort(class_accuracy)[:num_classes]

def build_per_class_dataloaders(
    dataset, 
    target_classes, 
    batch_size = 32
):
    """Returns a dict: {class_idx: DataLoader}"""
    class_dataloaders = {}
    for cls in target_classes:
        print(f'Building dataloader for class {cls}')
        indices = [i for i, (_, label) in enumerate(tqdm(dataset, desc = f'Class {cls}', unit = 'el')) if label == cls]
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        class_dataloaders[cls] = loader
    return class_dataloaders