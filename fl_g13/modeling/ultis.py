import os
from enum import Enum
import torch
import glob
import torch.optim as optim


class MODEL_DICTIONARY(Enum):
    EPOCH = 'epoch'
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'


def save_model(model, optimizer=None, checkpoint_dir=None, epoch=None, prefix_name="model"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if epoch is None:
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix_name}.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix_name}_epoch_{epoch}.pth")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    torch.save({
        MODEL_DICTIONARY.EPOCH.value: epoch,
        MODEL_DICTIONARY.MODEL_STATE_DICT.value: model.state_dict(),
        # MODEL_DICTIONARY.OPTIMIZER_STATE_DICT.value: optimizer.state_dict(),
    }, checkpoint_path)

    print(f"💾 Saved checkpoint at: {checkpoint_path}")


def load_or_create_model(checkpoint_dir=None, model=None, optimizer=None, lr=1e-4, weight_decay=0.04, device=None):
    if not checkpoint_dir and not model:
        raise ValueError("Either checkpoint_dir or model must be provided.")
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not checkpoint_dir:
        model.to(device)
        return model, optimizer, 1

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        model = vits16
        model.to(device)



    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pth")),
        key=os.path.getmtime
    )

    if checkpoint_files:
        # Load the latest checkpoint
        latest_ckpt = checkpoint_files[-1]
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint[MODEL_DICTIONARY.MODEL_STATE_DICT.value])
        # optimizer.load_state_dict(checkpoint[MODEL_DICTIONARY.OPTIMIZER_STATE_DICT.value])
        start_epoch = checkpoint.get(MODEL_DICTIONARY.EPOCH.value) or 0 + 1
        print(f"Loaded checkpoint from {latest_ckpt}, resuming at epoch {start_epoch}")
    else:
        start_epoch = 1
        print(f"No checkpoint found, initializing new model from scratch.")

    return model, optimizer, start_epoch
