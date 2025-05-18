from typing import Type
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim, device, cuda, nn, ones_like
from fl_g13.modeling.load import load_or_create, get_model
from fl_g13.editing import SparseSGDM
from fl_g13.architectures import BaseDino


class Net(nn.Module):
    """Model (simple CNN adapted for cifar100)"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Aggiunto pooling
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Aggiunto pooling
        # Calcola la dimensione corretta per l'input di fc1
        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # Modificato in base alla nuova dimensione
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # Riduce la dimensione spaziale
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # Riduce ulteriormente la dimensione spaziale
        x = x.view(x.size(0), -1)  # Appiattisce mantenendo la dimensione del batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TinyCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> [B, 16, 32, 32]
        x = F.max_pool2d(x, 2)  # -> [B, 16, 16, 16]
        x = F.relu(self.conv2(x))  # -> [B, 32, 16, 16]
        x = F.max_pool2d(x, 2)  # -> [B, 32, 8, 8]
        x = x.view(x.size(0), -1)  # -> [B, 32*8*8]
        x = self.fc1(x)  # -> [B, 100]
        return x

def get_default_model():
    return Net()

def get_experiment_setting(
        checkpoint_dir: str = None, 
        model_class: Type[nn.Module] | nn.Module = BaseDino, 
        learning_rate: float = 1e-3, 
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        model_editing: bool = False,
        model_config: dict = None,
        save_with_model_dir: bool = False,
    ):
    """Get the experiment setting."""
    dev = device("cuda:0" if cuda.is_available() else "cpu")
    model, start_epoch = load_or_create(
        path=f"{checkpoint_dir}/{model_class.__name__}" if save_with_model_dir else checkpoint_dir,
        model_class=model_class,
        model_config=model_config,
        device=dev,
        verbose=True,
    )
    if model_editing:
        mask = [ones_like(p, device = p.device) for p in model.parameters()] # Must be done AFTER the model is moved to the device
        optimizer = SparseSGDM(model.parameters(), mask=mask, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)
    return model, optimizer, criterion, dev, scheduler