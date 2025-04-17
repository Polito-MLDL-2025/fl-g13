from enum import Enum
import glob
import os

import torch

from fl_g13.modeling.utils import generate_goofy_name


class ModelKeys(Enum):
    EPOCH = "epoch"
    MODEL_STATE_DICT = "model_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    SCHEDULER_STATE_DICT = "scheduler_state_dict"


def save(checkpoint_dir, prefix, model, optimizer, scheduler=None, epoch=None):
    """Saves the model, optimizer, and optionally scheduler state to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not prefix:
        prefix = generate_goofy_name()

    if not epoch:
        filename = os.path.join(checkpoint_dir, f"{prefix}.pth")
    else:
        filename = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.pth")

    checkpoint = {
        ModelKeys.EPOCH.value: epoch,
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),
        ModelKeys.OPTIMIZER_STATE_DICT.value: optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value] = scheduler.state_dict()

    torch.save(checkpoint, filename)

    print(f"💾 Saved checkpoint at: {filename}")


def load(path, model, optimizer, scheduler=None, device=None):
    """
    Loads a checkpoint into the given model and optimizer (optionally scheduler). Raises an error if no checkpoint is found.
    """
    if os.path.isdir(path):
        checkpoint_files = sorted(
            glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime
        )
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {path}")
        ckpt_path = checkpoint_files[-1]
    elif os.path.isfile(path):
        ckpt_path = path
    else:
        raise FileNotFoundError(f"Checkpoint path is neither a file nor a directory: {path}")

    # TODO Could implement that if the path do not ends with _epoch_int then the most recent could be picked (higher epoch)

    if device:
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = torch.load(ckpt_path)

    model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])
    optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])
    if scheduler is not None and ModelKeys.SCHEDULER_STATE_DICT.value in checkpoint:
        scheduler.load_state_dict(checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value])

    start_epoch = checkpoint.get(ModelKeys.EPOCH.value, 0) + 1

    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")

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
