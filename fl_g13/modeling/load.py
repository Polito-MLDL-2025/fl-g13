from enum import Enum
import glob
import os

import torch

from fl_g13.modeling.utils import generate_goofy_name


class ModelKeys(Enum):
    EPOCH = 'epoch'
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
    SCHEDULER_STATE_DICT = 'scheduler_state_dict'


def save(checkpoint_dir, model, optimizer, scheduler=None, epoch=1, filename=None):
    """Saves the model, optimizer, and optionally scheduler state to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = generate_goofy_name()

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        ModelKeys.EPOCH.value: epoch,
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),
        ModelKeys.OPTIMIZER_STATE_DICT.value: optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint[ModelKeys.SCHEDULER_STATE_DICT.value] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)

    print(f"💾 Saved checkpoint at: {checkpoint_path}")

def load(checkpoint_dir, model, optimizer, scheduler=None, filename=None, device=None):
    """
    Loads the latest checkpoint into the given model and optimizer (optionally scheduler). Raises an error if no checkpoint is found.
    """
    if filename:
        ckpt_path = os.path.join(checkpoint_dir, filename)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Specified checkpoint file not found: {ckpt_path}")
    else:
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "*.pth")),
            key=os.path.getmtime
        )
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in directory: {checkpoint_dir}")
        ckpt_path = checkpoint_files[-1]

    if device:
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = torch.load(ckpt_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1

    print(f"✅ Loaded checkpoint from {ckpt_path}, resuming at epoch {start_epoch}")
    
    return start_epoch


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
