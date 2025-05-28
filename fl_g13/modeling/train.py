import time
from typing import Optional, List, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import save, save_loss_and_accuracy
from fl_g13.modeling.utils import generate_goofy_name


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    verbose: int = 1,
    num_steps: Optional[int] = None
) -> Tuple[float, float, List[float]]:
    """
    Train the model for a single epoch.

    Args:
        dataloader (DataLoader): DataLoader providing the training data.
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        verbose (int, optional): Verbosity level for progress display. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - float: Average training loss for the epoch.
            - float: Training accuracy for the epoch.
            - list: List of loss values for each iteration.
    """
    # Get the device where the model is located
    device = next(model.parameters()).device
    # Set the model to training mode
    model.train()

    # Set the for loop iterator according to the verbose flag
    if verbose == 1:
        # Default, use tqdm with progress bar
        batch_iterator = tqdm(dataloader, desc = 'Training progress', unit = 'batch')
    else:
        # No progress bar
        batch_iterator = dataloader

    # Initialize variables to track total loss, correct predictions, total samples, and per-iteration losses
    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []
    total_batches = len(dataloader)

    for batch_idx, (X, y) in enumerate(batch_iterator):

        # If num_steps is specified, stop training after reaching the limit
        print(f"Batch {batch_idx + 1}/{total_batches} | Batch size: {X.size(0)}")
        if num_steps is not None and batch_idx > num_steps:
            break

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

        # Verbose == 2 print progress every 10 batches, else every batch
        if verbose > 1 and (batch_idx + 1) % (10 if verbose == 2 else 1) == 0:
            print(f"  ↳ Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f}")
            
    # Compute the average training loss for the epoch
    training_loss = total_loss / total_batches
    # Compute the training accuracy for the epoch
    training_accuracy = correct / total
    return training_loss, training_accuracy, iteration_losses


def train(
    checkpoint_dir: Optional[str],         # Directory where model checkpoints will be saved
    name: Optional[str],         # Prefix or name for the model checkpoints
    start_epoch: int,            # Starting epoch number (useful for resuming training)
    num_epochs: int,             # Total number of epochs to train the model
    save_every: Optional[int],             # Frequency (in epochs) to save model checkpoints
    backup_every: Optional[int],           # Frequency (in epochs) to backup model checkpoints
    train_dataloader: DataLoader,       # DataLoader for the training dataset
    val_dataloader: Optional[DataLoader],         # DataLoader for the validation dataset
    model: Module,                              # The model to be trained
    criterion: Module,                          # Loss function used for training
    optimizer: Optimizer,                       # Optimizer used to update model parameters
    scheduler: Optional[_LRScheduler] = None,   # Learning rate scheduler (optional)
    verbose: int = 1,                      # Verbosity level
    all_training_losses: Optional[List[float]] = None,           # Pre-allocated list to store training losses (optional)
    all_validation_losses: Optional[List[float]] = None,         # Pre-allocated list to store validation losses (optional)
    all_training_accuracies: Optional[List[float]] = None,       # Pre-allocated list to store training accuracies (optional)
    all_validation_accuracies: Optional[List[float]] = None,     # Pre-allocated list to store validation accuracies (optional)
    eval_every: Optional[int] = 1,    #  Frequency (in epochs) to run evaluation model
    num_steps: Optional[int] = None,  # Number of steps to train the model (optional)
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model for a given number of epochs, periodically saving checkpoints and tracking performance metrics.
    
    Args:
        checkpoint_dir (str): Directory where model checkpoints will be saved.
        name (str): Prefix or name for the model checkpoints. If not provided, a random name will be generated.
        start_epoch (int): Starting epoch number (useful for resuming training).
        num_epochs (int): Total number of epochs to train the model.
        save_every (int): Frequency (in epochs) to save model checkpoints.
        backup_every (int): Frequency (in epochs) to backup model checkpoints.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler (optional). Defaults to None.
        verbose (int, optional): Verbosity level for progress display. Defaults to 1.
        all_training_losses (list, optional): Pre-allocated list to store training losses (optional). Defaults to None.
        all_validation_losses (list, optional): Pre-allocated list to store validation losses (optional). Defaults to None.
        all_training_accuracies (list, optional): Pre-allocated list to store training accuracies (optional). Defaults to None.
        all_validation_accuracies (list, optional): Pre-allocated list to store validation accuracies (optional). Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - all_training_losses (list): List of training losses for each epoch.
            - all_validation_losses (list): List of validation losses for each epoch.
            - all_training_accuracies (list): List of training accuracies for each epoch.
            - all_validation_accuracies (list): List of validation accuracies for each epoch.
    """

    # Generate a random prefix/name for the model if none is provided
    if not name:
        name = generate_goofy_name(checkpoint_dir)
        print(f"No prefix/name for the model was provided, choosen prefix/name: {name}")
        print()
    else:
        print(f"Prefix/name for the model was provided: {name}")
        print()

    # Initialize lists if not provided
    if all_training_losses is None:
        all_training_losses = []
    if val_dataloader and all_validation_losses is None:
        all_validation_losses = []
    if all_training_accuracies is None:
        all_training_accuracies = []
    if val_dataloader and all_validation_accuracies is None:
        all_validation_accuracies = []

    adjusted_end_epoch = start_epoch + num_epochs - 1
    for epoch in range(1, num_epochs + 1):
        
        # Adjust the epoch number for saving checkpoints
        adjusted_epoch = start_epoch + epoch - 1


        # Start the timer for the epoch
        start_time = time.time()

        # Train the model for one epoch
        train_loss, training_accuracy, _ = train_one_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            verbose=verbose,
            num_steps=num_steps
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
            f"🚀 Epoch {adjusted_epoch}/{adjusted_end_epoch} ({100 * adjusted_epoch / adjusted_end_epoch:.2f}%) Completed\n"
            f"\t📊 Training Loss: {train_loss:.4f}\n"
            f"\t✅ Training Accuracy: {100 * training_accuracy:.2f}%\n"
            f"\t⏳ Elapsed Time: {elapsed_time:.2f}s | ETA: {eta:.2f}s\n"
            f"\t🕒 Completed At: {current_time}"
        )
        if val_dataloader and eval_every and epoch % eval_every == 0:
            # Evaluate the model on the validation dataset
            validation_loss, validation_accuracy, _ = eval(
                dataloader=val_dataloader, model=model, criterion=criterion, verbose=verbose
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
            # Save the model, optimizer, and scheduler state
            save(checkpoint_dir=checkpoint_dir, prefix=name, model=model, epoch=adjusted_epoch, optimizer=optimizer, scheduler=scheduler)
            
            # Save the train/val loss and train/val accuracy
            train_epochs = list(range(start_epoch, adjusted_epoch + 1))
            val_epochs = list(range(start_epoch, adjusted_epoch + 1, eval_every)) if val_dataloader else None
            
            save_loss_and_accuracy(
                checkpoint_dir = checkpoint_dir,
                prefix = name,
                model = model,
                epoch = adjusted_epoch,
                train_losses = all_training_losses,
                train_accuracies = all_training_accuracies,
                train_epochs = train_epochs,
                val_losses = all_validation_losses,
                val_accuracies = all_validation_accuracies,
                val_epochs = val_epochs
            )
            print()

        if backup_every and epoch % backup_every == 0:
            # Backup the model, optimizer, and scheduler state to avoid overwriting
            print(f"Running backup for epoch {epoch}")
            save(checkpoint_dir=f"{checkpoint_dir}/backup", prefix=f"backup_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{name}", model=model, epoch=adjusted_epoch, optimizer=optimizer, scheduler=scheduler)
            
            # Backup metrics
            train_epochs = list(range(start_epoch, adjusted_epoch + 1))
            val_epochs = list(range(start_epoch, adjusted_epoch + 1, eval_every)) if val_dataloader else None
            save_loss_and_accuracy(
                checkpoint_dir = f"{checkpoint_dir}/backup",
                prefix = f"backup_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{name}",
                model = model,
                epoch = adjusted_epoch,
                train_losses = all_training_losses,
                train_accuracies = all_training_accuracies,
                train_epochs = train_epochs,
                val_losses = all_validation_losses,
                val_accuracies = all_validation_accuracies,
                val_epochs = val_epochs
            )
            print()


    return all_training_losses, all_validation_losses, all_training_accuracies, all_validation_accuracies
