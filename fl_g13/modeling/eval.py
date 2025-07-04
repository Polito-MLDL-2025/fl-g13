import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import List, Tuple
from tqdm import tqdm

def eval(
    dataloader: DataLoader,
    model: Module,
    criterion: Module,
    verbose: int = 1
) -> Tuple[float, float, List[float]]:
    """
    Evaluate the model's performance on a given dataloader using the specified loss function.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation dataset.
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): Loss function used for evaluation.
        verbose (int, optional): Verbosity level for progress display. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - test_loss (float): Average loss over all batches.
            - test_accuracy (float): Overall accuracy of the model on the evaluation dataset.
            - iteration_losses (list): List of per-batch losses.
    """
    device = next(model.parameters()).device
    model.eval()
    
    if verbose == 1:
        batch_iterator = tqdm(dataloader, desc = 'Eval progress', unit = 'batch')
    else:
        batch_iterator = dataloader # No progress bar

    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(batch_iterator):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item()
            iteration_losses.append(loss.item())
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == y).sum().item()
            batch_total = y.size(0)

            correct += batch_correct
            total += batch_total

            # Verbose == 2 print progress every 10 batches, else every batch
            if verbose > 1 and (batch_idx + 1) % (10 if verbose == 2 else 1) == 0:
                print(f"  ↳ Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    test_loss = total_loss / len(dataloader)
    test_accuracy = correct / total
    return test_loss, test_accuracy, iteration_losses