from typing import Type
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim, device, cuda, nn
from fl_g13.modeling.load import load_or_create, get_model


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


def get_default_model():
    return Net()

def get_experiment_setting(checkpoint_dir: str, model_class: Type[nn.Module] | nn.Module):
    """Get the experiment setting."""
    model = get_model(model_class)
    optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dev = device("cuda:0" if cuda.is_available() else "cpu")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=0.001)
    #model, _ = load_or_create(checkpoint_dir, model_class, dev, optimizer, scheduler)
    return model, optimizer, criterion, dev, scheduler