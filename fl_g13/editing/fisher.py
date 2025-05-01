import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from collections import defaultdict

from tqdm import tqdm


def fisher_scores(
    dataloader: DataLoader,
    model: Module,
    verbose: int = 1,
    loss_fn = torch.nn.CrossEntropyLoss()
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
        batch_iterator = tqdm(dataloader, desc = 'Per Class Accuracy', unit = 'batch')
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

def create_gradiend_mask(class_score, sparsity = 0.2):
    """
    class_score: dict of {param_name: tensor}
    sparsity: fraction of parameters to keep editable (lowest scores values)

    Returns: dict {param_name: binary_mask_tensor}
    """
    gradient_mask = {}

    for name, scores in class_score.items():
        scores_flat = scores.view(-1)
        k = int(len(scores_flat) * sparsity)

        if k == 0:
            # Prevent empty mask
            mask = torch.zeros_like(scores, dtype = torch.float)
        else:
            threshold, _ = torch.kthvalue(scores_flat, k)
            mask = (scores <= threshold).to(dtype = torch.float)

        gradient_mask[name] = mask

    return gradient_mask