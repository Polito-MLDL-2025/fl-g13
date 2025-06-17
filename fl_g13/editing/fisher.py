from collections import defaultdict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


def fisher_scores(
        dataloader: DataLoader,
        model: Module,
        verbose: int = 1,
        loss_fn=torch.nn.CrossEntropyLoss()
):
    """
    Computes diagonal Fisher Information scores for model parameters.
    Returns: {param_name: fisher_score_tensor}
    """
    # Get the device where the model is located
    device = next(model.parameters()).device
    # Set the model to evaluation mode    
    model.eval()

    # Set the for loop iterator according to the verbose flag
    if verbose == 1:
        # Default, use tqdm with progress bar
        batch_iterator = tqdm(dataloader, desc='Fisher Score', unit='batch')
    else:
        # No progress bar
        batch_iterator = dataloader

    # Initialize variables
    fisher_scores = defaultdict(lambda: 0)
    total_batches = len(dataloader)

    for batch_idx, (X, y) in enumerate(batch_iterator):
        X, y = X.to(device), y.to(device)

        model.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_scores[name] += (param.grad.detach() ** 2)

        # Verbose == 2 print progress every 10 batches
        if verbose == 2 and (batch_idx + 1) % 10 == 0:
            print(f"  ↳ Batch {batch_idx + 1}/{total_batches}")

    # Average over number of batches
    for name in fisher_scores:
        fisher_scores[name] /= total_batches

    return fisher_scores


def masked_fisher_score(
        dataloader,
        model,
        current_mask,
        verbose=1,
        loss_fn=torch.nn.CrossEntropyLoss()
):
    # Get the device where the model is located
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    if verbose > 0:
        print("Computing Masked Fisher Score")
        print(f"Device:{device}")
    # Set the model to evaluation mode    
    model.eval()

    # Set the for loop iterator according to the verbose flag
    if verbose == 1:
        # Default, use tqdm with progress bar
        batch_iterator = tqdm(dataloader, desc='Fisher Score', unit='batch')
    else:
        # No progress bar
        batch_iterator = dataloader

    # Initialize variables
    fisher_scores = defaultdict(lambda: 0)
    total_batches = len(dataloader)

    for name, param in model.named_parameters():
        if name not in current_mask:
            current_mask[name] = torch.ones_like(param.data).to(device)
        # Move mask to the correct device
        current_mask[name] = current_mask[name].to(device)

    for batch_idx, (X, y) in enumerate(batch_iterator):
        X, y = X.to(device), y.to(device)

        model.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                masked_grad = param.grad.detach() * current_mask[name].to(param.grad.device)
                fisher_scores[name] += (masked_grad ** 2)

        # Verbose == 2 print progress every 10 batches
        if verbose == 2 and (batch_idx + 1) % 10 == 0:
            print(f"  ↳ Batch {batch_idx + 1}/{total_batches}")

    # Average over number of batches
    for name in fisher_scores:
        fisher_scores[name] /= total_batches
    del loss, outputs  # help PyTorch clear references
    torch.cuda.empty_cache()
    return fisher_scores
