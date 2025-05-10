"""pytorch-example: A Flower / PyTorch app."""

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import json

from flwr.common.typing import UserConfig


def get_weights(net):
    """Get model weights as a list of ndarrays, one for each layer."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    # with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        # json.dump(config, fp)

    return save_path, run_dir