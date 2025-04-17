from enum import Enum
import glob
import os

import torch

from fl_g13.modeling.utils import generate_goofy_name


class ModelKeys(Enum):
    # Enum to define keys used in the checkpoint dictionary
    EPOCH = "epoch"
    MODEL_STATE_DICT = "model_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    SCHEDULER_STATE_DICT = "scheduler_state_dict"


def save(checkpoint_dir, prefix, model, optimizer, scheduler=None, epoch=None):
    """
    Saves the model, optimizer, and optionally scheduler state to a checkpoint file.

    Args:
        checkpoint_dir (str): Directory where the checkpoint file will be saved.
        prefix (str): Prefix for the checkpoint file name. If None, a random name will be generated.
        model (torch.nn.Module): The model whose state will be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler whose state will be saved. Defaults to None.
        epoch (int, optional): The current epoch number to include in the checkpoint file name. Defaults to None.

    Returns:
        None
    """
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Generate a prefix if none is provided
    if not prefix:
        prefix = generate_goofy_name()

    # Determine the filename based on whether an epoch is provided
    if not epoch:
        filename = os.path.join(checkpoint_dir, f"{prefix}.pth")
    else:
        filename = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.pth")

    # Create a dictionary to store the checkpoint data
    checkpoint = {
        ModelKeys.EPOCH.value: epoch,
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),
        ModelKeys.OPTIMIZER_STATE_DICT.value: optimizer.state_dict(),
    }

    # Add scheduler state to the checkpoint if provided
    if scheduler is not None:
        checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value] = scheduler.state_dict()

    # Save the checkpoint to the specified file
    torch.save(checkpoint, filename)

    # Print confirmation of the saved checkpoint
    print(f"💾 Saved checkpoint at: {filename}")


def load(path, model, optimizer, scheduler=None, device=None):
    """
    Loads a checkpoint into the provided model, optimizer, and optionally a scheduler. 
    Automatically determines whether the given path is a file (loads the specified file) 
    or a directory (loads the most recently modified checkpoint file in the directory).

    Raises:
        FileNotFoundError: If no checkpoint is found at the specified path.

    Args:
        path (str): Path to the checkpoint file or directory containing checkpoint files.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load the state into. Defaults to None.
        device (torch.device, optional): The device to map the checkpoint to. Defaults to None.

    Returns:
        int: The epoch to resume training from.
    """
    # Check if the path is a directory
    if os.path.isdir(path):
        # Get all checkpoint files in the directory, sorted by modification time
        checkpoint_files = sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime)
        # Raise an error if no checkpoint files are found
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {path}")
        # Use the most recent checkpoint file
        ckpt_path = checkpoint_files[-1]
    # Check if the path is a file
    elif os.path.isfile(path):
        ckpt_path = path
    # Raise an error if the path is neither a file nor a directory
    else:
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a directory: {path}")

    # TODO Could implement that if the path do not ends with _epoch_int then the most recent could be picked (higher epoch)

    # Load the checkpoint, optionally mapping it to a specific device
    if device:
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = torch.load(ckpt_path)

    # Load the model state from the checkpoint
    model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])
    # Load the optimizer state from the checkpoint
    optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])
    # Load the scheduler state from the checkpoint if provided and present in the checkpoint
    if scheduler is not None and ModelKeys.SCHEDULER_STATE_DICT.value in checkpoint:
        scheduler.load_state_dict(checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value])

    # Determine the starting epoch from the checkpoint, defaulting to 0 if not present
    start_epoch = checkpoint.get(ModelKeys.EPOCH.value, 0) + 1

    # Print confirmation of the loaded checkpoint and the resuming epoch
    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")

    # Return the starting epoch as an integer
    return int(start_epoch)


# def load_or_create_model(checkpoint_dir, model=None, optimizer=None, scheduler=None, lr=1e-4, weight_decay=0.04, device=None):
#     """Loads the latest checkpoint or initializes a new model, optimizer, and optionally scheduler."""
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if model is None:
#         model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)

#     if optimizer is None:
#         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")), key=os.path.getmtime)

#     if checkpoint_files:
#         latest_ckpt = checkpoint_files[-1]
#         checkpoint = torch.load(latest_ckpt, map_location=device)
#         model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])
#         optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])
#         start_epoch = checkpoint[ModelKeys.EPOCH.value] + 1

#         if scheduler is not None and ModelKeys.SCHEDULER_STATE_DICT.value in checkpoint:
#             scheduler.load_state_dict(checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value])

#         print(f"✅ Loaded checkpoint from {latest_ckpt}, resuming at epoch {start_epoch}")
#     else:
#         start_epoch = 1
#         print("⚠️ No checkpoint found, initializing new model from scratch.")

#     return model, optimizer, scheduler, start_epoch
