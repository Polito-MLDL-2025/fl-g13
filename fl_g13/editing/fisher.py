from collections import defaultdict
from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

def fisher_scores(
    dataloader: DataLoader,
    model: Module,
    verbose: int = 1,
    loss_fn: Module = torch.nn.CrossEntropyLoss()
):
    """
    Computes the Fisher information score for each parameter in the model.

    The Fisher score is approximated as the diagonal of the Fisher Information
    Matrix, which is calculated as the average of the squared gradients of the
    loss with respect to the parameters, over all samples in the dataloader.

    Args:
        dataloader (DataLoader): DataLoader for the dataset to compute scores on.
        model (Module): The model to evaluate.
        verbose (int, optional): Verbosity level for progress display.
            0: silent, 1: tqdm progress bar, 2: detailed batch logs.
            Defaults to 1.
        loss_fn (Module, optional): The loss function to use for backpropagation.
            Defaults to torch.nn.CrossEntropyLoss().

    Returns:
        dict[str, torch.Tensor]: A dictionary where keys are parameter names and
            values are the corresponding Fisher scores.
    """
    device = next(model.parameters()).device
    model.eval()

    if verbose == 1:
        batch_iterator = tqdm(dataloader, desc='Fisher Score', unit='batch')
    else:        
        batch_iterator = dataloader # No progress bar

    scores = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    for batch_idx, (X, y) in enumerate(batch_iterator):
        X, y = X.to(device), y.to(device)

        model.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                scores[name] += (param.grad.detach() ** 2)

        if verbose == 2 and (batch_idx + 1) % 10 == 0:
            print(f"  ↳ Batch {batch_idx + 1}/{len(dataloader)}")

    # Average over number of batches
    for name in scores:
        scores[name] /= len(dataloader)

    return scores

def masked_fisher_score(
    dataloader: DataLoader,
    model: Module,
    current_mask: Dict[str, torch.Tensor],
    verbose: int = 1,
    loss_fn: Module = torch.nn.CrossEntropyLoss()
) -> Dict[str, torch.Tensor]:
    """
    Computes Fisher scores on a model's parameters, applying a mask to the gradients.

    This is similar to `fisher_scores`, but it multiplies the gradients by a
    provided `current_mask` before squaring and accumulating them.

    If a parameter name from the model is not found in `current_mask`, a default
    mask of all ones is created and used for that parameter.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        model (Module): The model to evaluate.
        current_mask (Dict[str, torch.Tensor]): A dictionary mapping parameter names
            to binary mask tensors.
        verbose (int, optional): Verbosity level. 0: silent, 1: tqdm progress bar,
            2: detailed batch logs. 
            Defaults to 1.
        loss_fn (Module, optional): The loss function to use.
            Defaults to torch.nn.CrossEntropyLoss().

    Returns:
        Dict[str, torch.Tensor]: A dictionary of masked Fisher scores.
    """
    device = next(model.parameters()).device
    model.eval()

    if verbose == 1:
        batch_iterator = tqdm(dataloader, desc='Masked Fisher Score', unit='batch')
    else:
        batch_iterator = dataloader

    scores = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name not in current_mask:
                current_mask[name] = torch.ones_like(param.data)
            current_mask[name] = current_mask[name].to(device)

    for batch_idx, (X, y) in enumerate(batch_iterator):
        X, y = X.to(device), y.to(device)

        model.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                masked_grad = param.grad.detach() * current_mask[name]
                scores[name] += masked_grad ** 2

        if verbose > 1 and (batch_idx + 1) % 10 == 0:
            print(f"  ↳ Batch {batch_idx + 1}/{len(dataloader)}")

    num_batches = len(dataloader)
    if num_batches > 0:
        for name in scores:
            scores[name] /= num_batches

    return scores