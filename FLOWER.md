# Flower Implementation Overview

This document outlines the federated learning architecture implemented using the Flower framework.
The implementation is structured around Flower's core abstractions: `ClientApp`, `ServerApp`, and `Strategy`.
All files can be found under `fl_g13\fl_pytorch`.

## Core Components

The FL system is divided into server-side and client-side components, each with distinct responsibilities.

### 1. Server-Side Implementation

The server orchestrates the FL process, aggregates model updates, and manages the overall training flow.

#### `server_app.py`
This file acts as the server factory. The `get_server_app` function is the main entry point for configuring and creating the `ServerApp`. It initializes the global model, evaluation functions, and the chosen server-side strategy.

#### Strategies
Strategies define the core logic of the federated learning process, including how clients are selected, how their contributions are aggregated, and how the global model is updated.

- **`strategy.py` (`CustomFedAvg`)**
  - This is the base strategy, extending Flower's `FedAvg`.
  - **Responsibilities**:
    - Standard weighted averaging of client model parameters.
    - Periodic server-side model checkpointing.
    - Logging metrics (training loss, accuracy, etc.) to Weights & Biases.
    - Calculating and logging the average L2 norm of the drift between client parameters and the global model.
    - Performing periodic centralized evaluation on the server test set.

- **`DynamicQuorumStrategy.py` (`DynamicQuorum`)**
  - This is our prposed strategy AdaQuo that inherits from `CustomFedAvg`.
  - **Key Features**:
    - **Global Mask Generation**: It computes a global vote tensor based on the sum of individual client masks and a `quorum` threshold. Only parameters present in at least `quorum` number of client masks are kept.
    - **Quorum Update**: The `current_quorum` can be updated dynamically during training using two modes:
      1.  **Linear**: Increases the quorum by a fixed `quorum_increment` at a regular `quorum_update_frequency`.
      2.  **Adaptive**: Increases the quorum only when the average client `drift` is below a `drift_threshold`, indicating model stability. It also includes a patience mechanism and a forced update rule.
    - It sends the computed global mask to clients via the `configure_fit` method's configuration dictionary.

### 2. Client-Side Implementation

The client-side logic defines how each participant handles its local data and model.

#### `client_app.py`
This file is the client factory. The `get_client_app` function creates the `ClientApp`, which in turn uses a `client_fn` to instantiate a specific client for each simulation participant. It is responsible for loading the client's partitioned data.

#### Client Implementations

- **`client.py` (`CustomNumpyClient`)**
  - This is the base client, inheriting from Flower's `NumPyClient`.
  - **Responsibilities**:
    - Receives global model parameters from the server.
    - Performs local training for a set number of epochs or steps.
    - **Model Editing**: If `model_editing` is enabled, it can compute a local sparsity mask based on the approx. of the FIM from its own data. This mask is applied to its optimizer to constrain local training.
    - Can save and load its mask and model weights to/from its `client_state` to persist them across rounds.
    - Returns updated local weights and training metrics to the server.

- **`DynamicQuorumClient.py` (`DynamicQuorumClient`)**
  - This client is designed to work with the `DynamicQuorum` server strategy.
  - It inherits from `CustomNumpyClient` but overrides the `fit` method to handle the global mask sent by the server.
  - When it receives a `global_mask` in the configuration from the server, it applies this mask to its optimizer, overriding any local mask computation.

### 3. Supporting Modules

- **`datasets.py`**: Provides functions (`load_flwr_datasets`, `get_transforms`) to load, partition (IID or non-IID), and transform the dataset (e.g., CIFAR-100) for each client.
- **`editing/centralized_mask.py`**: Contains tools to pre-compute client masks and scores. This is crucial for the `DynamicQuorum` strategy, which requires the sum of all client masks to be available at initialization.
- **`task.py`**: Includes utility functions (`get_weights`, `set_weights`) for converting between PyTorch model state dictionaries and Flower's `NDArrays` format.
- **`model.py`**: Defines a simple model architecture  `TinyCNN` that can be used for debugging.
- **`utils.py`**: Helper functions for downloading external dependencies required for specific model architectures (e.g., from the DINO repository).

## Execution Workflow

A federated learning run follows these steps:

1.  **Initialization**: A main code block configures the experiment, initializes the model and dependencies, and calls `get_server_app` and `get_client_app`.
2.  **Simulation Start**: `flwr.simulation.run_simulation` launches the server and the pool of clients.
3.  **Server Setup**: The `server_fn` within `get_server_app` runs once, creating the specified `Strategy` (`CustomFedAvg` or `DynamicQuorum`).
4.  **Federated Rounds**:
    - **Server (Selection & Configuration)**: The strategy's `configure_fit` method is called.
    - **Client (Instantiation)**: The `client_fn` is called for each selected client, creating an instance of `CustomNumpyClient` or `DynamicQuorumClient`.
    - **Client (Training)**: The client's `fit` method is executed. It receives the global parameters and the configuration and performs local training.
    - **Server (Aggregation)**: The strategy's `aggregate_fit` method collects and aggregates the results.
5.  **Loop**: Step 4 is repeated for the specified number of rounds.
