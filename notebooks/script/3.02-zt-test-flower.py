#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


from fl_g13.config import RAW_DATA_DIR


from torchvision import datasets, transforms

from fl_g13.base_experimentation import dataset_handler

import torch
import torch.nn as nn
import torch.nn.functional as F


import flwr
from flwr.common import Context

from flwr.simulation import run_simulation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()


# # Load data

transform = transforms.Compose([
    transforms.ToTensor()
])
cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=transform)


### train val split
train_dataset,val_dataset = dataset_handler.train_test_split(cifar100_train,train_ratio=0.8)


# I.I.D Sharding Split
## k client
k =10
clients_dataset_train= dataset_handler.iid_sharding(train_dataset,k)
clients_dataset_val= dataset_handler.iid_sharding(val_dataset,k)


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


# ## Init model , optimizer and loss function

net = TinyCNN().to(DEVICE)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.04)
criterion = torch.nn.CrossEntropyLoss()


# # Define the ClientApp

# ## Build module local
# 
# Build module local such that ClientApp can use it

get_ipython().system('pip install -e ..')


# ## create FlowerClient instances  

'''
Function load data client is to simulate the distribution data into each client
In the real case, each client will have its dataset
'''
def load_data_client(context: Context):
    partition_id = context.node_config["partition-id"] 
    print(f"Client {partition_id} is ready to train")
    trainloader = DataLoader(clients_dataset_train[partition_id])
    valloader = DataLoader(clients_dataset_val[partition_id])
    return trainloader, valloader


# ### Create instant of ClientApp

from fl_g13.fl_pytorch.client_app import get_client_app

client = get_client_app(load_data_client,model=net,optimizer=optimizer,criterion=criterion,device=DEVICE)


# # Define the Flower ServerApp
# 
# Customize built-in strategy Federated Averaging (FedAvg) of Flower to combine hyperparams in server-side and save model for each k epoch
# 
# The strategy could also incremental training an

# ## Create instant of ServerApp

from pathlib import Path
from torch.utils.data import DataLoader
from fl_g13.fl_pytorch.server_app import get_server_app

def get_datatest_fn(context: Context):
    return DataLoader(cifar100_test)

## checkpoints directory
current_path = Path.cwd()
model_test_path = current_path / "../models/model_test"
model_test_path.resolve()


num_rounds=2
save_every =2
fraction_fit=1.0  # Sample 100% of available clients for training
fraction_evaluate=0.5  # Sample 50% of available clients for evaluation
min_fit_clients=10  # Never sample less than 10 clients for training
min_evaluate_clients=5  # Never sample less than 5 clients for evaluation
min_available_clients=10  # Wait until all 10 clients are available
device=DEVICE
use_wandb=False


server = get_server_app(checkpoint_dir=model_test_path.resolve(),
                        model=net,optimizer=optimizer,criterion=criterion, get_datatest_fn=get_datatest_fn,
                        num_rounds=num_rounds,
                        fraction_fit=fraction_fit, 
                        fraction_evaluate=fraction_evaluate,  
                        min_fit_clients=min_fit_clients,  
                        min_evaluate_clients=min_evaluate_clients, 
                        min_available_clients=min_available_clients, 
                        device=device,
                        use_wandb=use_wandb,
                        save_every=save_every
                        )


# # Run the training
# 

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config["client_resources"]= {"num_cpus": 1, "num_gpus": 0.25}
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




