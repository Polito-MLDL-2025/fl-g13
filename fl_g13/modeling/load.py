from enum import Enum
import glob
import os
from typing import Optional, Tuple, Type, List, Dict
import json

import torch
from torch import nn, optim
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

from fl_g13.modeling.utils import generate_goofy_name


class ModelKeys(Enum):
    EPOCH = "epoch"
    MODEL_STATE_DICT = "model_state_dict"
    MODEL_CLASS = "model_class"
    CONFIG = "config"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    OPTIMIZER_CLASS = "optimizer_class"
    SCHEDULER_STATE_DICT = "scheduler_state_dict"
    SCHEDULER_CLASS = "scheduler_class"


def save(
        checkpoint_dir: str,
        prefix: Optional[str],
        model: nn.Module,
        epoch: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        filename: Optional[str] = None,
        with_model_dir: bool = True,
) -> None:
    """
    Saves the model state to a checkpoint file under a subfolder named after the model's class name.

    Args:
        checkpoint_dir (str): Directory where the checkpoint file will be saved.
        prefix (Optional[str]): Prefix for the checkpoint file name. If None, a random name will be generated.
        model (torch.nn.Module): The model whose state will be saved.
        epoch (int): The current epoch number to include in the checkpoint file name.
        optimizer (Optional[Optimizer]): The optimizer to save, if provided.
        scheduler (Optional[_LRScheduler]): The learning rate scheduler to save, if provided.
        filename (Optional[str]): set filename for save file if provided
        with_model_dir
    Returns:
        None
    """
    # Get the name of the model class
    model_name = model.__class__.__name__

    # Create a directory for saving checkpoints specific to the model
    if with_model_dir:
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If no prefix is provided, generate a random one
    if not prefix:
        prefix = generate_goofy_name()

    # Construct the filename for the checkpoint
    if not filename:
        filename = os.path.join(checkpoint_dir, f"{prefix}_{model_name}_epoch_{epoch}.pth")
    else:
        filename = os.path.join(checkpoint_dir, filename)
    # Create a dictionary to store the checkpoint data
    checkpoint = {
        ModelKeys.EPOCH.value: epoch,  # Save the current epoch
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),  # Save model parameters
        ModelKeys.MODEL_CLASS.value: model.__class__.__name__,  # Save model class name
    }
    if hasattr(model, "_config") and model._config is not None:
        checkpoint[ModelKeys.CONFIG.value] = model._config  # Save model configuration

    # If optimizer is provided, save its state and class name
    if optimizer is not None:
        checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value] = optimizer.state_dict()
        checkpoint[ModelKeys.OPTIMIZER_CLASS.value] = optimizer.__class__.__name__

    # If scheduler is provided, save its state and class name
    if scheduler is not None:
        checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value] = scheduler.state_dict()
        checkpoint[ModelKeys.SCHEDULER_CLASS.value] = scheduler.__class__.__name__

    # Save the checkpoint dictionary to the specified file
    torch.save(checkpoint, filename)

    # Print confirmation message with the path to the saved checkpoint
    print(f"💾 Saved checkpoint at: {filename}")

def save_loss_and_accuracy(
    checkpoint_dir: str, 
    prefix: Optional[str],
    model: nn.Module,
    epoch: int,
    train_losses: List[float],
    train_accuracies: List[float],
    train_epochs: List[int],
    val_losses: Optional[List[float]],
    val_accuracies: Optional[List[float]],
    val_epochs: Optional[List[int]],
    filename: Optional[str] = None,
    with_model_dir: bool = True,
):
    """
    Saves the training loss and accuracy and the validation loss and accuracy to a json file under 
        a subfolder named after the model's class name.

    Args:
        checkpoint_dir (str): Directory where the json file will be saved.
        prefix (Optional[str]): Prefix for the json file name. If None, a random name will be generated.
        model (torch.nn.Module): The model whose metrics will be saved.
        epoch (int): The current epoch number to include in the json file name.
        train_losses (List[float]): Training losses values.
        train_accuracies (List[float]): Training accuracies values.
        train_epochs (List[int]): Epoch number for each training measurement.
        val_losses (Optional[List[float]]): Validation losses values.
        val_accuracies (Optional[List[float]]): Validation accuracies values.
        val_epochs (Optional[List[int]]): Epoch number for each validation measurement.
        filename (Optional[str]): set filename for save file if provided
        with_model_dir
    Returns:
        None
    """
    # Get the name of the model class
    model_name = model.__class__.__name__

    # Create a directory for saving checkpoints specific to the model
    if with_model_dir:
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If no prefix is provided, generate a random one
    if not prefix:
        prefix = generate_goofy_name()

    # Construct the filename for the checkpoint
    if not filename:
        filename = os.path.join(checkpoint_dir, f"{prefix}_{model_name}_epoch_{epoch}.loss_acc.json")
    else:
        filename = os.path.join(checkpoint_dir, filename)
    
    # Create a dictionary with the data and store in a json file 
    metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'train_epochs': train_epochs,
        'val_epochs': val_epochs 
    }
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent = 4)
        
    # Print confirmation message with the path to the saved checkpoint
    print(f"💾 Saved losses and accuracies (training and validation) at: {filename}")

def load(
        path: str,
        model_class: Type[Module]| Module,
        device: Optional[torch.device] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        verbose: bool = False
) -> Tuple[nn.Module, int]:
    """
    Loads a checkpoint into a new model and optionally restores optimizer and scheduler.

    Args:
        path (str): Path to the checkpoint file or directory containing checkpoint files.
        model_class (Type[Module]|Module): The class used to instantiate the model.
        device (Optional[torch.device]): The device to map the checkpoint to.
        optimizer (Optional[Optimizer]): Optimizer instance to load state into, if provided.
        scheduler (Optional[_LRScheduler]): Scheduler instance to load state into, if provided.
        verbose (bool): Whether to print loading info.
    Returns:
        Tuple[nn.Module, int]: The model with restored state and the epoch to resume training from.
    """
    # Check if the path is a directory
    if os.path.isdir(path):
        # Get all .pth files in the directory, sorted by modification time (oldest to newest)
        checkpoint_files = sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime)

        # Raise an error if no checkpoint files are found
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {path}")

        # Use the most recent checkpoint file
        ckpt_path = checkpoint_files[-1]

    # If path is a file, use it directly as the checkpoint path
    elif os.path.isfile(path):
        ckpt_path = path

    # If path is neither file nor directory, raise an error
    else:
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a directory: {path}")

    # Load the checkpoint, mapping it to the specified device if given
    checkpoint = torch.load(ckpt_path, map_location=device) if device else torch.load(ckpt_path)

    # If verbose is enabled, print checkpoint details for debugging
    if verbose:
        print(f"🔍 Loading checkpoint from {ckpt_path}")

        # Print model class info if available
        if ModelKeys.MODEL_CLASS.value in checkpoint:
            print(f"📦 Model class in checkpoint: {checkpoint[ModelKeys.MODEL_CLASS.value]}")

        # Print optimizer class info if available
        if ModelKeys.OPTIMIZER_CLASS.value in checkpoint:
            print(f"⚙️ Optimizer class in checkpoint: {checkpoint[ModelKeys.OPTIMIZER_CLASS.value]}")

        # Print scheduler class info if available
        if ModelKeys.SCHEDULER_CLASS.value in checkpoint:
            print(f"📈 Scheduler class in checkpoint: {checkpoint[ModelKeys.SCHEDULER_CLASS.value]}")

        # Print model configuration info if available
        if ModelKeys.CONFIG.value in checkpoint:
            print(f"🔧 Model configuration: {checkpoint[ModelKeys.CONFIG.value]}")

    model = get_model(model_class)

    if hasattr(model_class, "from_config") and callable(getattr(model_class, "from_config")):
        model = model_class.from_config(checkpoint[ModelKeys.CONFIG.value])

    # Load model weights from the checkpoint
    model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])

    # Load optimizer state if an optimizer is provided and the checkpoint contains its state
    if optimizer and ModelKeys.OPTIMIZER_STATE_DICT.value in checkpoint:
        optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])

    # Load scheduler state if a scheduler is provided and the checkpoint contains its state
    if scheduler and ModelKeys.SCHEDULER_STATE_DICT.value in checkpoint:
        scheduler.load_state_dict(checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value])

    # Determine the epoch to resume from (one after the saved epoch, default to 0 if not found)
    start_epoch = checkpoint.get(ModelKeys.EPOCH.value, 0) + 1

    # Confirm successful loading
    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")

    # Return the loaded model and starting epoch
    return model, start_epoch


def load_or_create(
        path: str,
        model_class: Type[nn.Module] | Module,
        model_config = None,
        device: Optional[torch.device] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        verbose: bool = False
) -> tuple[Module, int]:
    """
    Loads a model checkpoint or creates a new model if no checkpoint is found.
    This function attempts to load a checkpoint from the specified path into a model. 
    If no checkpoint is found, it creates a new model instance. Optionally, it can also 
    restore the states of an optimizer and a learning rate scheduler.
        model_class (Type[nn.Module] | nn.Module): The model class or an instance of the model.
        model_config (Optional[dict]): Configuration dictionary for creating the model, 
            used if no checkpoint is found. If provided, the model class must have a 
            `from_config` method.
        device (Optional[torch.device]): The device to map the model and checkpoint to. 
            Defaults to "cuda:0" if available, otherwise "cpu".
        optimizer (Optional[torch.optim.Optimizer]): Optimizer instance to restore state into, if provided.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler 
            instance to restore state into, if provided.
        verbose (bool): If True, prints information about the loading process. Defaults to False.
        tuple[nn.Module, int]: A tuple containing the model (with restored state if a checkpoint 
        was found) and the epoch to resume training from. If no checkpoint is found, the epoch 
        is set to 1.
    Raises:
        ValueError: If a model configuration is provided but the model class does not have 
        a `from_config` method.
    """
    try:
        return load(path, model_class, device, optimizer, scheduler, verbose)
    except FileNotFoundError:
        if verbose:
            print(f"⚠️ No checkpoint found at {path}. Creating a new model.")

        if not model_config:
            model = get_model(model_class)
        elif hasattr(model_class, 'from_config'):
            model = model_class.from_config(model_config)
        else:
            raise ValueError(f"You provided a model config but no method to load such was found in model_class {model_class}")
        
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device:
            model.to(device)

        return model, 1

def load_loss_and_accuracies(
    path: str,
    verbose: bool = False
)-> Dict[str, Optional[List[float|int]]]:
    """
    Loads the training losses and accuracies and validation losses and accuracies from a file.

    Args:
        path (str): Path to the checkpoint file.
        verbose (bool): Whether to print loading info.
    Returns:
        Dict[str, Optional[List[float|int]]]: Dictionary with the metrics. 
    """
    # Check if the file exists
    if not os.path.isfile(path):
        # raise an error if the file does not exists
        raise FileNotFoundError(f"Specified path does not exists: {path}")
    
    # Load data
    with open(path, 'r') as f:
        loaded_metrics = json.load(f)
    
    print(f"📈 Loss and Accuracy data correctly loaded")
    if verbose:
        print("\tKeys found:", list(loaded_metrics.keys()))
        
    return loaded_metrics

def get_model(model_class: Type[nn.Module] | Module):
    # If model_class is already an instance, use it directly
    if isinstance(model_class, nn.Module):
        model = model_class
    else:
        # Otherwise, instantiate a new model with a default config
        try:
            model = model_class()
        except TypeError as e:
            raise ValueError("Model class requires a config or arguments for instantiation.") from e
    return model

import matplotlib.pyplot as plt

def plot_metrics(
    path: str
):
    # load metrics
    metrics = load_loss_and_accuracies(path)

    train_epochs = metrics['train_epochs']
    val_epochs = metrics.get('val_epochs', [])

    # two plots, separate for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(train_epochs, metrics['train_loss'], label='Train Loss', color='tab:blue')
    if metrics['val_loss']:
        ax1.plot(val_epochs, metrics['val_loss'], label='Val Loss', color='tab:orange')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(train_epochs, metrics['train_acc'], label='Train Accuracy', color='tab:green')
    if metrics['val_acc']:
        ax2.plot(val_epochs, metrics['val_acc'], label='Val Accuracy', color='tab:red')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
