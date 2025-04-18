from enum import Enum
import glob
import os
from typing import Optional, Tuple, Type

import torch
from torch import nn, optim
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
    scheduler: Optional[_LRScheduler] = None
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

    Returns:
        None
    """
    model_name = model.__class__.__name__
    model_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    if not prefix:
        prefix = generate_goofy_name()

    filename = os.path.join(model_dir, f"{prefix}_{model_name}_epoch_{epoch}.pth")

    checkpoint = {
        ModelKeys.EPOCH.value: epoch,
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),
        ModelKeys.CONFIG.value: model._config,
        ModelKeys.MODEL_CLASS.value: model.__class__.__name__,
    }

    if optimizer is not None:
        checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value] = optimizer.state_dict()
        checkpoint[ModelKeys.OPTIMIZER_CLASS.value] = optimizer.__class__.__name__

    if scheduler is not None:
        checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value] = scheduler.state_dict()
        checkpoint[ModelKeys.SCHEDULER_CLASS.value] = scheduler.__class__.__name__

    torch.save(checkpoint, filename)
    print(f"💾 Saved checkpoint at: {filename}")


def load(
    path: str,
    model_class: Type[nn.Module],
    device: Optional[torch.device] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    verbose: bool = False
) -> Tuple[nn.Module, int]:
    """
    Loads a checkpoint into a new model and optionally restores optimizer and scheduler.

    Args:
        path (str): Path to the checkpoint file or directory containing checkpoint files.
        model_class (Type[nn.Module]): The class used to instantiate the model.
        device (Optional[torch.device]): The device to map the checkpoint to.
        optimizer (Optional[Optimizer]): Optimizer instance to load state into, if provided.
        scheduler (Optional[_LRScheduler]): Scheduler instance to load state into, if provided.

    Returns:
        Tuple[nn.Module, int]: The model with restored state and the epoch to resume training from.
    """
    if os.path.isdir(path):
        checkpoint_files = sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime)
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {path}")
        ckpt_path = checkpoint_files[-1]
    elif os.path.isfile(path):
        ckpt_path = path
    else:
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a directory: {path}")

    checkpoint = torch.load(ckpt_path, map_location=device) if device else torch.load(ckpt_path)

    # Print content of the checkpoint for debugging
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

    model = model_class.from_config(checkpoint[ModelKeys.CONFIG.value])
    model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])

    if optimizer and ModelKeys.OPTIMIZER_STATE_DICT.value in checkpoint:
        optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])

    if scheduler and ModelKeys.SCHEDULER_STATE_DICT.value in checkpoint:
        scheduler.load_state_dict(checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value])

    start_epoch = checkpoint.get(ModelKeys.EPOCH.value, 0) + 1
    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")

    return model, start_epoch
