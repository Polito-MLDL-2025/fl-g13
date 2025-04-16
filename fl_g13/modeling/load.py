from enum import Enum
import glob
import os

import torch
import torch.optim as optim

from fl_g13.modeling.utils import generate_goofy_name


class ModelKeys(Enum):
    EPOCH = 'epoch'
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'


def save_model(checkpoint_dir, model, optimizer, scheduler, epoch=1, filename=generate_goofy_name()):
    """Saves the model and optimizer state to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        ModelKeys.EPOCH.value: epoch,
        ModelKeys.MODEL_STATE_DICT.value: model.state_dict(),
        ModelKeys.OPTIMIZER_STATE_DICT.value: optimizer.state_dict(),
    }, checkpoint_path)

    print(f"💾 Saved checkpoint at: {checkpoint_path}")


def load_or_create_model(checkpoint_dir, model=None, optimizer=None, lr=1e-4, weight_decay=0.04, device=None):
    """Loads the latest checkpoint or initializes a new model and optimizer."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")), key=os.path.getmtime)

    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint[ModelKeys.MODEL_STATE_DICT.value])
        optimizer.load_state_dict(checkpoint[ModelKeys.OPTIMIZER_STATE_DICT.value])
        start_epoch = checkpoint[ModelKeys.EPOCH.value] + 1
        print(f"✅ Loaded checkpoint from {latest_ckpt}, resuming at epoch {start_epoch}")
    else:
        start_epoch = 1
        print("⚠️ No checkpoint found, initializing new model from scratch.")

    return model, optimizer, start_epoch
