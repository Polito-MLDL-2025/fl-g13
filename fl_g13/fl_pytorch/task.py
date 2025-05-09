from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
from flwr.common.typing import UserConfig

def get_weights(model): #? May this lead to CUDA incompatibility?
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
