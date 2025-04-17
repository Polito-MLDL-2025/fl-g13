import time

import torch

from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import save
from fl_g13.modeling.utils import generate_goofy_name


def train_one_epoch(dataloader, model, criterion, optimizer, verbose=False):
    """
    Trains the model for one epoch using the provided dataloader, optimizer, and loss function.
    Returns the average training loss, training accuracy, and a list of loss values for each iteration.
    """
    # Get the device where the model is located
    device = next(model.parameters()).device
    # Set the model to training mode
    model.train()

    # Initialize variables to track total loss, correct predictions, total samples, and per-iteration losses
    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []

    for batch_idx, (X, y) in enumerate(dataloader):
        # Move input data and labels to the same device as the model
        X, y = X.to(device), y.to(device)
        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # Perform a forward pass through the model
        logits = model(X)
        # Compute the loss using the criterion
        loss = criterion(logits, y)
        # Backpropagate the loss
        loss.backward()
        # Update the model's parameters
        optimizer.step()

        # Accumulate the total loss
        total_loss += loss.item()
        # Append the current loss to the iteration losses list
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

        # Print progress every 10 batches if verbose is enabled
        if verbose and batch_idx % 10 == 0:
            print(f"  ↳ Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    # Compute the average training loss for the epoch
    training_loss = total_loss / len(dataloader)
    # Compute the training accuracy for the epoch
    training_accuracy = correct / total
    return training_loss, training_accuracy, iteration_losses


def train(
    checkpoint_dir,
    prefix,
    start_epoch,
    num_epochs,
    save_every,
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    scheduler=None,
    verbose=False,
    all_training_losses=None,  # Pre-allocated list for training losses
    all_validation_losses=None,  # Pre-allocated list for validation losses
    all_training_accuracies=None,  # Pre-allocated list for training accuracies
    all_validation_accuracies=None,  # Pre-allocated list for validation accuracies
):
    """
    Trains a model for a specified number of epochs, saving checkpoints periodically.
    """

    # Generate a random prefix/name for the model if none is provided
    if not prefix:
        prefix = generate_goofy_name(checkpoint_dir)
        print(f"No prefix/name for the model was provided, choosen prefix/name: {prefix}")
        print()

    # Initialize lists if not provided
    if all_training_losses is None:
        all_training_losses = []
    if all_validation_losses is None:
        all_validation_losses = []
    if all_training_accuracies is None:
        all_training_accuracies = []
    if all_validation_accuracies is None:
        all_validation_accuracies = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train the model for one epoch
        train_loss, training_accuracy, _ = train_one_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            verbose=verbose,
        )
        # Append the per-iteration training losses and accuracy to the total lists
        all_training_losses.append(train_loss)
        all_training_accuracies.append(training_accuracy)

        # Calculate elapsed time and estimate time remaining
        elapsed_time = time.time() - start_time
        eta = elapsed_time * (num_epochs - epoch)

        # Print training results for the current epoch
        current_time = time.strftime("%H:%M", time.localtime())
        print(
            f"🚀 Epoch {epoch}/{num_epochs} ({100 * epoch / num_epochs:.2f}%) Completed\n"
            f"\t📊 Training Loss: {train_loss:.4f}\n"
            f"\t✅ Training Accuracy: {100 * training_accuracy:.2f}%\n"
            f"\t⏳ Elapsed Time: {elapsed_time:.2f}s | ETA: {eta:.2f}s\n"
            f"\t🕒 Completed At: {current_time}"
        )

        # Evaluate the model on the validation dataset
        validation_loss, validation_accuracy, _ = eval(
            dataloader=val_dataloader, model=model, criterion=criterion
        )
        # Append the per-iteration validation losses and accuracy to the total lists
        all_validation_losses.append(validation_loss)
        all_validation_accuracies.append(validation_accuracy)

        # Print validation results for the current epoch
        print(
            f"🔍 Validation Results:\n"
            f"\t📉 Validation Loss: {validation_loss:.4f}\n"
            f"\t🎯 Validation Accuracy: {100 * validation_accuracy:.2f}%"
        )

        print()

        # Update the learning rate scheduler if provided
        if scheduler:
            scheduler.step()

        # Save the model checkpoint periodically based on save_every
        if save_every and epoch % save_every == 0:
            # Calculate the saving epoch number
            saving_epoch = start_epoch + epoch - 1
            # Save the model, optimizer, and scheduler state
            save(checkpoint_dir, prefix, model, optimizer, scheduler, epoch=saving_epoch)
            print()

    return all_training_losses, all_validation_losses, all_training_accuracies, all_validation_accuracies
