from enum import Enum
import glob
import os
from typing import Optional, Tuple, Type

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
        device: Optional[torch.device] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        verbose: bool = False
) -> tuple[Module, int]:
    """
    Loads a checkpoint into a new model and optionally restores optimizer and scheduler.
    If no checkpoint is found, creates a new model and saves it.

    Args:
        path (str): Path to the checkpoint file or directory containing checkpoint files.
        model_class (Type[nn.Module] | nn.Module): Model class or instance.
        device (Optional[torch.device]): The device to map the checkpoint to.
        optimizer (Optional[Optimizer]): Optimizer instance to load state into, if provided.
        scheduler (Optional[_LRScheduler]): Scheduler instance to load state into, if provided.
        verbose (bool): Whether to print loading info.
    Returns:
        Tuple[nn.Module, int]: The model with restored state and the epoch to resume training from.
    """
    try:
        return load(path, model_class, device, optimizer, scheduler, verbose)
    except FileNotFoundError:
        if verbose:
            print(f"⚠️ No checkpoint found at {path}. Creating a new model.")

        model = get_model(model_class)

        if device:
            model.to(device)

        return model, 1


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
