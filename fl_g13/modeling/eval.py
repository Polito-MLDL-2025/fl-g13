import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import List, Tuple
from tqdm import tqdm

def eval(
    dataloader: DataLoader,                      # DataLoader providing the evaluation dataset
    model: Module,                               # The model to evaluate
    criterion: Module,                           # Loss function used for evaluation
    verbose: int = 1                        # If > 0, display progress
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
    # Get the device where the model is located
    device = next(model.parameters()).device
    # Set the model to evaluation mode
    model.eval()
    
        # Set the for loop iterator according to the verbose flag
    if verbose == 1:
        # Default, use tqdm with progress bar
        batch_iterator = tqdm(dataloader, desc = 'Eval progress', unit = 'batch')
    else:
        # No progress bar
        batch_iterator = dataloader

    # Initialize variables to track total loss, correct predictions, total samples, and per-batch losses
    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []
    total_batches = len(dataloader)
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches in the dataloader
        for batch_idx, (X, y) in enumerate(batch_iterator):
            # Move input data and labels to the same device as the model
            X, y = X.to(device), y.to(device)
            # Perform a forward pass through the model
            logits = model(X)
            # Compute the loss for the current batch
            loss = criterion(logits, y)

            # Accumulate the loss
            total_loss += loss.item()
            # Append the current batch loss to the list
            iteration_losses.append(loss.item())
            # Get the predicted class labels
            _, predicted = torch.max(logits, 1)
            # Count the number of correct predictions in the batch
            batch_correct = (predicted == y).sum().item()
            # Get the total number of samples in the batch
            batch_total = y.size(0)

            # Update the total correct predictions and total samples
            correct += batch_correct
            total += batch_total

            # Verbose == 2 print progress every 10 batches
            if verbose == 2 and (batch_idx + 1) % 10 == 0:
                print(f"  ↳ Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f}")

    # Compute the average loss over all batches
    test_loss = total_loss / total_batches
    # Compute the overall accuracy
    test_accuracy = correct / total

    # Return the average loss, accuracy, and per-batch losses
    return test_loss, test_accuracy, iteration_losses
