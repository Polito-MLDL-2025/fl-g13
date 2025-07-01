from enum import Enum
import glob
import os
from typing import Optional, Tuple, Type, List, Dict
import json

import torch
from torch import nn, optim
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.pyplot as plt

from fl_g13.modeling.utils import generate_unique_name

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
        filename (Optional[str]): The filename for the checkpoint file, if provided.
        with_model_dir (bool): Whether to save the checkpoint in a subdirectory named after the model's class.
    Returns:
        None
    """
    model_name = model.__class__.__name__

    # Create a directory for saving checkpoints specific to the model
    if with_model_dir:
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not prefix:
        prefix = generate_unique_name()

    # Construct the filename for the checkpoint
    if not filename:
        filename = os.path.join(checkpoint_dir, f"{prefix}_{model_name}_epoch_{epoch}.pth")
    else:
        filename = os.path.join(checkpoint_dir, filename)

    # Create a dictionary to store the checkpoint data
    checkpoint = {
        ModelKeys.EPOCH.value: epoch,                           # current epoch
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),   # model parameters
        ModelKeys.MODEL_CLASS.value: model.__class__.__name__,  # model class name
    }
    if hasattr(model, "_config") and model._config is not None:
        checkpoint[ModelKeys.CONFIG.value] = model._config      # model configuration

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
    model_name = model.__class__.__name__

    # Create a directory for saving checkpoints specific to the model
    if with_model_dir:
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not prefix:
        prefix = generate_unique_name()

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
    if os.path.isdir(path):
        # Get all .pth files in the directory, sorted by modification time (oldest to newest)
        checkpoint_files = sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime)
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {path}")

        ckpt_path = checkpoint_files[-1] # most recent file
    elif os.path.isfile(path):
        ckpt_path = path
    else:
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a directory: {path}")

    checkpoint = torch.load(ckpt_path, map_location=device) if device else torch.load(ckpt_path)

    if verbose:
        print(f"🔍 Loading checkpoint from {ckpt_path}")
        if ModelKeys.MODEL_CLASS.value in checkpoint:
            print(f"📦 Model class in checkpoint: {checkpoint[ModelKeys.MODEL_CLASS.value]}")
        if ModelKeys.OPTIMIZER_CLASS.value in checkpoint:
            print(f"⚙️ Optimizer class in checkpoint: {checkpoint[ModelKeys.OPTIMIZER_CLASS.value]}")
        if ModelKeys.SCHEDULER_CLASS.value in checkpoint:
            print(f"📈 Scheduler class in checkpoint: {checkpoint[ModelKeys.SCHEDULER_CLASS.value]}")
        if ModelKeys.CONFIG.value in checkpoint:
            print(f"🔧 Model configuration: {checkpoint[ModelKeys.CONFIG.value]}")

    model = get_model(model_class)
    if hasattr(model_class, "from_config") and callable(getattr(model_class, "from_config")):
        model = model_class.from_config(checkpoint[ModelKeys.CONFIG.value])
    model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])

    if device:
        model.to(device)
        if verbose:
            print(f"➡️ Moved model to device: {device}")

    if optimizer and ModelKeys.OPTIMIZER_STATE_DICT.value in checkpoint:
        optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])

    if scheduler and ModelKeys.SCHEDULER_STATE_DICT.value in checkpoint:
        scheduler.load_state_dict(checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value])

    start_epoch = checkpoint.get(ModelKeys.EPOCH.value, 0) + 1
    
    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")
    return model, start_epoch

def load_or_create(
        path: str,
        model_class: Type[nn.Module] | Module,
        model_config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        verbose: bool = False
) -> tuple[Module, int]:
    """
    Attempts to load a model checkpoint from a given path; creates a new model if no checkpoint is found.

    This function acts as a convenient wrapper around `load`. If a checkpoint exists at the specified
    path, it will be loaded. If the path does not exist or contains no checkpoint files, a new
    instance of the model will be created, initialized, and moved to the specified device.

    Args:
        path (str): The path to the checkpoint file or directory.
        model_class (Type[nn.Module] | Module): The class of the model to load or create.
        model_config (Optional[Dict], optional): Configuration dictionary used to instantiate a new model
            via a `from_config` class method if the checkpoint is not found. Defaults to None.
        device (Optional[torch.device], optional): The device to map the model to. If None, defaults to
            "cuda" if available, otherwise "cpu". Defaults to None.
        optimizer (Optional[optim.Optimizer], optional): An optimizer instance whose state will be
            loaded from the checkpoint if available. Defaults to None.
        scheduler (Optional[_LRScheduler], optional): A scheduler instance whose state will be
            loaded from the checkpoint if available. Defaults to None.
        verbose (bool, optional): If True, prints detailed loading information or a warning if a new
            model is created. Defaults to False.

    Returns:
        Tuple[Module, int]: A tuple containing:
            - The loaded or newly created model instance.
            - The starting epoch number (from the checkpoint, or 1 for a new model).
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
        
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Specified path does not exists: {path}")
    
    with open(path, 'r') as f:
        loaded_metrics = json.load(f)
    
    print(f"📈 Loss and Accuracy data correctly loaded")
    if verbose:
        print("\tKeys found:", list(loaded_metrics.keys()))
        
    return loaded_metrics

def get_model(model_class: Type[nn.Module] | Module):
    if isinstance(model_class, nn.Module):
        model = model_class
    else:
        try:
            model = model_class()
        except TypeError as e:
            raise ValueError("Model class requires a config or arguments for instantiation.") from e
    return model

def plot_metrics(
    path: str
):
    metrics = load_loss_and_accuracies(path)

    train_epochs = metrics['train_epochs']
    val_epochs = metrics.get('val_epochs', [])

    # two plots, loss and accuracy
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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