from enum import Enum
import glob
import os
from typing import Optional, Tuple, Type

import torch
from torch import nn

from fl_g13.modeling.utils import generate_goofy_name


class ModelKeys(Enum):
    EPOCH = "epoch"
    MODEL_STATE_DICT = "model_state_dict"
    CONFIG = "config"  # Removed unused entries


def save(checkpoint_dir: str, prefix: Optional[str], model: nn.Module, epoch: int) -> None:
    """
    Saves the model state to a checkpoint file.

    Args:
        checkpoint_dir (str): Directory where the checkpoint file will be saved.
        prefix (Optional[str]): Prefix for the checkpoint file name. If None, a random name will be generated.
        model (torch.nn.Module): The model whose state will be saved.
        epoch (int): The current epoch number to include in the checkpoint file name.

    Returns:
        None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not prefix:
        prefix = generate_goofy_name()

    model_name = model.__class__.__name__
    filename = os.path.join(checkpoint_dir, f"{prefix}_{model_name}_epoch_{epoch}.pth")

    checkpoint = {
        ModelKeys.EPOCH.value: epoch,
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),
        ModelKeys.CONFIG.value: model._config,  # Save model config as an enum entry
    }

    torch.save(checkpoint, filename)
    print(f"💾 Saved checkpoint at: {filename}")


def load(path: str, model_class: Type[nn.Module], device: Optional[torch.device] = None) -> Tuple[nn.Module, int]:
    """
    Loads a checkpoint into a new model, and restores the model state. 
    The optimizer and scheduler are re-initialized based on the current configuration.

    Args:
        path (str): Path to the checkpoint file or directory containing checkpoint files.
        model_class (Type[torch.nn.Module]): The class used to instantiate the model.
        device (Optional[torch.device]): The device to map the checkpoint to. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, int]: The model with restored state and the epoch to resume training from.
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

    # Load model state
    model = model_class.from_config(checkpoint[ModelKeys.CONFIG.value])  # Use enum for config
    model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])

    start_epoch = checkpoint.get(ModelKeys.EPOCH.value, 0) + 1
    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")

    return model, start_epoch