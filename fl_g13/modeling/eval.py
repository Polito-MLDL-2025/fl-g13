import torch


def eval(dataloader, model, criterion, verbose=False):
    """
    Evaluate the model on the given dataloader using the specified loss function.
    If verbose is True, print progress and intermediate results.
    """
    # Get the device where the model is located
    device = next(model.parameters()).device
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track total loss, correct predictions, total samples, and per-batch losses
    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches in the dataloader
        for batch_idx, (X, y) in enumerate(dataloader):
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

            # If verbose is enabled, print progress every 10 batches
            if verbose and batch_idx % 10 == 0:
                print(f"  ↳ Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    # Compute the average loss over all batches
    test_loss = total_loss / len(dataloader)
    # Compute the overall accuracy
    test_accuracy = correct / total

    # Return the average loss, accuracy, and per-batch losses
    return test_loss, test_accuracy, iteration_losses
