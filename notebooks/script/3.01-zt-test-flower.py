#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install -U ipywidgets')
get_ipython().system('pip install -q flwr[simulation] flwr-datasets[vision] torch torchvision matplotlib')


import sys
import os
current_directory = os.getcwd()
current_directory
sys.path.append(current_directory+'\\..\\fl_g13')


from fl_g13.modeling.test import test_model
from fl_g13.modeling.train import train_model
from fl_g13.config import RAW_DATA_DIR
from collections import OrderedDict
from typing import List


import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from collections import Counter

from fl_g13.base_experimentation import dataset_handler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()


# # Load data

transform = transforms.Compose([
    transforms.ToTensor()
])
cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)


### train val split
train_dataset,val_dataset = dataset_handler.train_test_split(cifar100_train,train_ratio=0.8)


# I.I.D Sharding Split
## k client
k =10
clients_dataset_train= dataset_handler.iid_sharding(train_dataset,k)
clients_dataset_val= dataset_handler.iid_sharding(val_dataset,k)


dataset_handler.check_subset_distribution(clients_dataset_train[2])


# # Update model parameters

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# # Define the ClientApp

from enum import Enum
import glob
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
import typer



app = typer.Typer()

class MODEL_DICTIONARY(Enum):
    EPOCH = 'epoch'
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'


def save_model(model, optimizer, checkpoint_dir, epoch=None, prefix_name="model"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if epoch is None:
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix_name}.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"{prefix_name}_epoch_{epoch}.pth")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    torch.save({
        MODEL_DICTIONARY.EPOCH.value: epoch,
        MODEL_DICTIONARY.MODEL_STATE_DICT.value: model.state_dict(),
        MODEL_DICTIONARY.OPTIMIZER_STATE_DICT.value: optimizer.state_dict(),
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
        optimizer.load_state_dict(checkpoint[MODEL_DICTIONARY.OPTIMIZER_STATE_DICT.value])
        start_epoch = checkpoint[MODEL_DICTIONARY.EPOCH.value] + 1
        print(f"Loaded checkpoint from {latest_ckpt}, resuming at epoch {start_epoch}")
    else:
        start_epoch = 1
        print(f"No checkpoint found, initializing new model from scratch.")

    return model, optimizer, start_epoch


def _train(model, optimizer, dataloader, loss_fn, device, is_print=False):
    # TODD
    # print("Training...")
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch, (X, y) in enumerate(dataloader):
        # print(f"Batch {batch+1}/{len(dataloader)}")
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        batch_acc = 100 * (predicted == y).sum().item() / y.size(0)
        if is_print:
            print(f"  ↳ Batch {batch + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.2f}%")

    training_loss = total_loss / len(dataloader)
    training_accuracy = 100 * correct / total
    print(f"Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%")
    return training_loss, training_accuracy


def train_model(checkpoint_dir, dataloader, loss_fn=torch.nn.CrossEntropyLoss(),
                num_epochs=10, save_every=None, lr=1e-4, weight_decay=0.04,
                model=None, optimizer=None, device=None, print_batch=False):

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, optimizer, start_epoch = load_or_create_model(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay
    )
    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss, training_accuracy = _train(model, optimizer, dataloader, loss_fn, device, print_batch)
        print(f"📘 Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}, Accuracy: {training_accuracy:.2f}%")

        # 5. Save checkpoint
        if save_every and epoch % save_every == 0:
            save_model(model, optimizer, checkpoint_dir, epoch, prefix_name="dino_xcit")



def test_model(model, dataloader, loss_fn,device=None):
    # TODO
    model.eval()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total_loss = 0.0
    correct = 0
    total = 0
    preds = []
    labels = []
    probs = []
    inputs = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)


            inputs.extend(X.cpu())
            total_loss += loss.item()
            original_pre = pred.cpu()
            prob, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            preds.extend(predicted.cpu())
            labels.extend(y.cpu())
            probs.extend(prob.cpu())

    test_loss = total_loss / len(dataloader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    probs = torch.tensor(probs)
    return preds, labels, probs, inputs, test_accuracy,test_loss


class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader,loss_fn=torch.nn.CrossEntropyLoss(),optimizer=None):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.loss_fn = loss_fn
        if not optimizer:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.04)
        self.optimizer = optimizer

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_model(checkpoint_dir=None,dataloader= self.trainloader, num_epochs=1,save_every=None,model=self.model,optimizer=self.optimizer,loss_fn=self.loss_fn,print_batch=False)
        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        preds, labels, probs, inputs, test_accuracy,test_loss = test_model(self.model, self.valloader, self.loss_fn)
        return float(test_loss), len(self.valloader), {"accuracy": float(test_accuracy)}


# ## Tiny model

class TinyCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # -> [B, 16, 32, 32]
        x = F.max_pool2d(x, 2)        # -> [B, 16, 16, 16]
        x = F.relu(self.conv2(x))     # -> [B, 32, 16, 16]
        x = F.max_pool2d(x, 2)        # -> [B, 32, 8, 8]
        x = x.view(x.size(0), -1)     # -> [B, 32*8*8]
        x = self.fc1(x)               # -> [B, 100]
        return x


# ## create FlowerClient instances  

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = TinyCNN().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    
    trainloader = torch.utils.data.DataLoader(clients_dataset_train[partition_id])
    valloader = torch.utils.data.DataLoader(clients_dataset_val[partition_id])
    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)


# # Define the Flower ServerApp
# 
# Using a built-in strategy Federated Averaging (FedAvg) of Flower to combine hyperparams in server-side

from flwr.common import FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from typing import Union, Optional

net = TinyCNN().to(DEVICE)
class SaveModelStrategy(flwr.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = flwr.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics



# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)


# ## Create instant of ServerApp

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=4)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)


# # Run the training
# 

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# 

NUM_CLIENTS =10


# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)




