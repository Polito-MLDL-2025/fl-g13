import time
import itertools
from typing import Optional, List, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import save, save_loss_and_accuracy
from fl_g13.modeling.utils import generate_unique_name

def train_for_steps(
    dataloader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    verbose: int = 1,
    num_steps: Optional[int] = None
) -> Tuple[float, float, List[float]]:
    """
    Train the model for a specified number of steps, which may span across multiple epochs.

    This function is useful for scenarios where training is defined by a fixed number of updates
    rather than a full pass over the dataset. If the dataloader is exhausted before all steps
    are completed, it will be re-initialized to continue training.

    Args:
        dataloader (DataLoader): DataLoader providing the training data.
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        verbose (int, optional): Verbosity level for progress display. Defaults to 1.
        num_steps (Optional[int]): The total number of training steps to perform.

    Returns:
        tuple: A tuple containing:
            - float: Average training loss over the performed steps.
            - float: Training accuracy over the performed steps.
            - list: List of loss values for each iteration/step.

    Raises:
        ValueError: If `num_steps` is not provided (is None).
    """
    if num_steps is None:
        raise ValueError("num_steps must be provided for train_for_steps.")
    if num_steps == 0:
        return 0.0, 0.0, []

    device = next(model.parameters()).device
    model.train()

    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []

    data_iter = itertools.cycle(dataloader)

    # Setup iterator based on verbosity
    if verbose == 1:
        iterator = tqdm(range(num_steps), desc='Training for steps', unit='step')
    else:
        iterator = range(num_steps)

    for step in iterator:
        X, y = next(data_iter)
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        total_loss += loss_item
        iteration_losses.append(loss_item)

        _, predicted = torch.max(logits, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

        if verbose == 1:
            iterator.set_postfix(loss=f"{loss_item:.4f}", acc=f"{(correct / total):.2%}")

        if verbose > 1 and ((step + 1) % (10 if verbose == 2 else 1) == 0):
            print(f"  ↳ Step {step + 1}/{num_steps} | Loss: {loss_item:.4f} | Accuracy: {(correct / total):.2%}")

    training_loss = total_loss / num_steps
    training_accuracy = correct / total
    return training_loss, training_accuracy, iteration_losses

def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    verbose: int = 1,
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
    device = next(model.parameters()).device
    model.train()

    if verbose == 1:
        batch_iterator = tqdm(dataloader, desc='Training progress', unit='batch')
    else:
        batch_iterator = dataloader

    total_loss, correct, total = 0.0, 0, 0
    iteration_losses = []

    for batch_idx, (X, y) in enumerate(batch_iterator):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        total_loss += loss_item
        iteration_losses.append(loss_item)

        _, predicted = torch.max(logits, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

        if verbose == 1:
            batch_iterator.set_postfix(loss=f"{loss_item:.4f}", acc=f"{(correct / total):.2%}")

        if verbose > 1 and (batch_idx + 1) % (10 if verbose == 2 else 1) == 0:
            print(
                f"  ↳ Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss_item:.4f} | Accuracy: {(correct / total):.2%}"
            )
            
    training_loss = total_loss / len(dataloader)
    training_accuracy = correct / total
    return training_loss, training_accuracy, iteration_losses

def train(
    checkpoint_dir: Optional[str],
    name: Optional[str],
    start_epoch: int,
    num_epochs: int,
    save_every: Optional[int],
    backup_every: Optional[int],
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    verbose: int = 1,
    all_training_losses: Optional[List[float]] = None,
    all_validation_losses: Optional[List[float]] = None,
    all_training_accuracies: Optional[List[float]] = None,
    all_validation_accuracies: Optional[List[float]] = None,
    eval_every: Optional[int] = 1,
    num_steps: Optional[int] = None,
    with_model_dir: Optional[bool] = True
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model, with periodic evaluation, checkpointing, and backup.

    Can train for a fixed number of epochs or a fixed number of steps.
    
    Args:
        checkpoint_dir (Optional[str]): Directory to save model checkpoints.
        name (Optional[str]): A name for the training run. If None, a unique name is generated.
        start_epoch (int): The starting epoch number, useful for resuming training.
        num_epochs (int): The total number of epochs to train for.
        save_every (Optional[int]): Frequency (in epochs) to save model checkpoints.
        backup_every (Optional[int]): Frequency (in epochs) to create a backup checkpoint.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (Optional[DataLoader]): DataLoader for the validation dataset.
        model (Module): The model to be trained.
        criterion (Module): The loss function.
        optimizer (Optimizer): The optimizer for updating model parameters.
        scheduler (Optional[_LRScheduler], optional): Learning rate scheduler. Defaults to None.
        verbose (int, optional): Verbosity level. 0=silent, 1=progress bars, >1=detailed logs. Defaults to 1.
        all_training_losses (Optional[List[float]], optional): List to append training losses. Defaults to None.
        all_validation_losses (Optional[List[float]], optional): List to append validation losses. Defaults to None.
        all_training_accuracies (Optional[List[float]], optional): List to append training accuracies. Defaults to None.
        all_validation_accuracies (Optional[List[float]], optional): List to append validation accuracies. Defaults to None.
        eval_every (Optional[int], optional): Frequency (in epochs) to run evaluation. Defaults to 1.
        num_steps (Optional[int], optional): If specified, train for a fixed number of steps instead of epochs. Defaults to None.
        with_model_dir (Optional[bool], optional): Whether to save checkpoints in a subdirectory named after the model. Defaults to True
    
    Returns:
        tuple: A tuple containing:
            - all_training_losses (list): List of training losses for each epoch.
            - all_validation_losses (list): List of validation losses for each epoch.
            - all_training_accuracies (list): List of training accuracies for each epoch.
            - all_validation_accuracies (list): List of validation accuracies for each epoch.
    """

    # Generate a random prefix/name for the model if none is provided
    if not name:
        name = generate_unique_name(checkpoint_dir)
        if verbose > 0:
            print(f"No prefix/name for the model was provided, choosen prefix/name: {name}")
    else:
        if verbose > 0:
            print(f"Prefix/name for the model was provided: {name}")

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
        
        start_time = time.time()
        if num_steps is not None:
            # Train the model for a specified number of steps
            train_loss, training_accuracy, _ = train_for_steps(
                dataloader=train_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                verbose=verbose,
                num_steps=num_steps
            )
        else:
            # Train the model for one epoch
            train_loss, training_accuracy, _ = train_one_epoch(
                dataloader=train_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                verbose=verbose,
            )
        elapsed_time = time.time() - start_time
        eta = elapsed_time * (num_epochs - epoch)
        
        # Append the per-iteration training losses and accuracy to the total lists
        all_training_losses.append(train_loss)
        all_training_accuracies.append(training_accuracy)

        # Print training results for the current epoch
        current_time = time.strftime("%H:%M", time.localtime())
        if verbose > 0:
            print(
                f"🚀 Epoch {adjusted_epoch}/{adjusted_end_epoch} ({100 * adjusted_epoch / adjusted_end_epoch:.2f}%) Completed\n"
                f"\t📊 Training Loss: {train_loss:.4f}\n"
                f"\t✅ Training Accuracy: {100 * training_accuracy:.2f}%\n"
                f"\t⏳ Elapsed Time: {elapsed_time:.2f}s | ETA: {eta:.2f}s\n"
                f"\t🕒 Completed At: {current_time}"
            )
        
        # Validation round, if needed
        if val_dataloader and eval_every and epoch % eval_every == 0:
            validation_loss, validation_accuracy, _ = eval(
                dataloader=val_dataloader, 
                model=model, 
                criterion=criterion, 
                verbose=verbose
            )
            all_validation_losses.append(validation_loss)
            all_validation_accuracies.append(validation_accuracy)

            if verbose > 0:
                print(
                    f"🔍 Validation Results:\n"
                    f"\t📉 Validation Loss: {validation_loss:.4f}\n"
                    f"\t🎯 Validation Accuracy: {100 * validation_accuracy:.2f}%"
                )

        if scheduler:
            scheduler.step()

        # Save the model checkpoint periodically
        if save_every and epoch % save_every == 0:
            # Save the model, optimizer, and scheduler state
            save(
                checkpoint_dir=checkpoint_dir, 
                prefix=name, 
                model=model, 
                epoch=adjusted_epoch, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                with_model_dir=with_model_dir
            )
            
            # Save the train/val loss and train/val accuracy
            train_epochs = list(range(start_epoch, adjusted_epoch + 1))
            val_epochs = list(range(start_epoch, adjusted_epoch + 1, eval_every)) if val_dataloader else None
            
            save_loss_and_accuracy(
                checkpoint_dir=checkpoint_dir,
                prefix=name,
                model=model,
                epoch=adjusted_epoch,
                train_losses=all_training_losses,
                train_accuracies=all_training_accuracies,
                train_epochs=train_epochs,
                val_losses=all_validation_losses,
                val_accuracies=all_validation_accuracies,
                val_epochs=val_epochs,
                with_model_dir= with_model_dir
            )

        if backup_every and epoch % backup_every == 0:
            if verbose > 0:
                print(f"Running backup for epoch {epoch}")
            save(
                checkpoint_dir=f"{checkpoint_dir}/backup", 
                prefix=f"backup_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{name}", 
                model=model, 
                epoch=adjusted_epoch, 
                optimizer=optimizer, 
                scheduler=scheduler
            )
            
            train_epochs = list(range(start_epoch, adjusted_epoch + 1))
            val_epochs = list(range(start_epoch, adjusted_epoch + 1, eval_every)) if val_dataloader else None
            save_loss_and_accuracy(
                checkpoint_dir=f"{checkpoint_dir}/backup",
                prefix=f"backup_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{name}",
                model=model,
                epoch=adjusted_epoch,
                train_losses=all_training_losses,
                train_accuracies=all_training_accuracies,
                train_epochs=train_epochs,
                val_losses=all_validation_losses,
                val_accuracies=all_validation_accuracies,
                val_epochs=val_epochs
            )
            
    return all_training_losses, all_validation_losses, all_training_accuracies, all_validation_accuracies