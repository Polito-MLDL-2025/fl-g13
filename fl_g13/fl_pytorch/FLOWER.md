# How Flower Works

This project implements a federated learning setup using the Flower framework with PyTorch. This section describes the core components and flow of the Flower implementation within this specific project.

Flower is used here to orchestrate the federated learning process, managing communication and coordination between a central server and multiple simulated clients.

The implementation is structured around the following 3 core Flower abstractions: `ClientApp`, `ServerApp`, and `Strategy`.

## Client, Server, and Strategy

### Client-Side Implementation (`client_app.py`)

The client-side logic is defined in `client_app.py`. It describes what each individual participant (client) does during the federated learning rounds.

- **`FlowerClient` Class:**
  - This class inherits from `flwr.client.NumPyClient`. It defines how a client interacts with the server and performs local machine learning tasks.
  - `__init__(...)`: Initializes the client with its local data loaders (`trainloader`, `valloader`), the model, optimizer, criterion, scheduler, device, and configuration for local training (`local_epochs`). It also handles the setup for optional `model_editing` (masking) and accesses the client's persistent `client_state`.
  - `_compute_mask(...)`: A custom method that computes a sparsity mask based on Fisher scores calculated on the client's local validation data. This mask is applied to the client's optimizer.
  - `set_mask(...)`: Applies a given mask (potentially received from the server) to the client's optimizer.
  - `_save_weights_to_state()`: Saves the client's current model state dictionary into the client's persistent `client_state` under the key `"full_model_state"`. This is intended for personalization but is currently saved without being reloaded in `fit`/`evaluate`.
  - `_load_weights_from_state()`: Loads a previously saved model state dictionary from the client's `client_state`. This is the counterpart to `_save_weights_to_state` but is currently commented out in `fit`/`evaluate`.
  - `fit(parameters, config)`: This is the main training method called by the server. It receives the global model parameters, applies them to the local model, performs local training for `local_epochs`, calculates client drift, saves the model state (if enabled), and returns the updated local model parameters, the number of training examples, and metrics (including the computed mask if `model_editing` is true).
  - `evaluate(parameters, config)`: This method is called by the server for federated evaluation. It receives the global model parameters, applies them, evaluates the model on the client's local validation data, and returns the evaluation loss, the number of validation examples, and metrics (accuracy).

- **`load_client_dataloaders(...)`**: A helper function that takes the Flower `Context` (containing client ID and partition info) and configuration to load the appropriate data partition for a specific client using `load_flwr_datasets`.
- **`get_client_app(...)`**: This function acts as a factory. It takes configuration parameters and defines the `client_fn`. The `client_fn` is called by the Flower simulation engine for each client to create and return a `FlowerClient` instance, configured with its data and the provided model/training components. It returns a `flwr.client.ClientApp` instance.

### Server-Side Implementation (`server_app.py`)

The central server logic is defined in `server_app.py`. This code runs on the central server orchestrating the federated learning rounds.

- **`get_evaluate_fn(...)`**: A utility function that creates the server-side evaluation function. This function is called by the server strategy to evaluate the aggregated global model on a centralized test set (loaded within the `server_fn`).
- **`fit_metrics_aggregation_fn(...)`**: Defines how metrics returned by clients' `fit` methods (like `train_loss` and `drift`) are aggregated on the server.
- **`evaluate_metrics_aggregation_fn(...)`**: Defines how metrics returned by clients' `evaluate` methods (like `accuracy`) are aggregated on the server.
- **`get_server_app(...)`**: This function acts as the server factory. It takes configuration parameters, loads or creates the initial global model using `load_or_create`, defines the `server_fn`, and creates a `flwr.server.ServerApp`.
  - **`server_fn(context)`**: This inner function is called by the Flower simulation engine once to set up the server environment. It loads the centralized test dataset, creates the `evaluate_fn`, retrieves initial model parameters, defines the server `strategy` (`MaskedFedAvg`), sets up the `ServerConfig` (number of rounds), and returns a `flwr.server.ServerAppComponents` object.

### Server Strategy (`strategy.py`)

The aggregation and server-side logic that runs each round is defined in `strategy.py`. This class extends a base Flower strategy to customize the aggregation process and add features like saving and logging.

- **`CustomFedAvg` Class:** This class inherits from `flwr.server.strategy.FedAvg`, extending the standard Federated Averaging algorithm with custom features.
  - `__init__(...)`: Initializes the strategy with the server's model, checkpoint directory, saving frequency, and W&B logging options. It calls the parent `FedAvg` constructor to inherit its core aggregation logic.
  - `aggregate_fit(server_round, results, failures)`: This method is called after clients complete their `fit` phase.
    - It calls `super().aggregate_fit(...)` to perform the standard weighted averaging of the client parameters.
    - It logs aggregated metrics (like average drift) and optionally saves the aggregated global model checkpoint.
  - `aggregate_evaluate(server_round, results, failures)`: Aggregates metrics from clients' `evaluate` calls using the `evaluate_metrics_aggregation_fn`.
  - `evaluate(server_round, parameters)`: Runs the centralized server-side evaluation using the `evaluate_fn` defined in `server_app.py`. It receives the current global parameters (which have been potentially masked by `aggregate_fit`).
  - `_init_wandb_project()`: Initializes the Weights & Biases logging run (if enabled).
  - `store_results_and_log(...)`**: Helper to log metrics to W&B or store them locally (though local storing is currently inactive).

> **Note**: if you need to make a personalized strategy, extend the `CustomFedAvg` class with your own implementation

### Simulation Execution (e.g., `run.py` or Jupyter Notebook)

The simulation is launched from a main script or notebook.

- It sets up experiment configuration parameters (number of clients, rounds, local epochs, batch size, etc.).
- It initializes the central model, optimizer, criterion, and determines the target `device` (CPU or CUDA).
- It calls `get_client_app(...)` and `get_server_app(...)` to get the configured `ClientApp` and `ServerApp` instances.
- It configures the `backend_config`, crucially setting `num_gpus` for `client_resources` when running on CUDA to ensure client processes can access the GPU during simulation.
- It calls `flwr.simulation.run_simulation(client_app=..., server_app=..., num_supernodes=..., backend_config=...)` to start the simulation.

### Dependencies

- The project relies on external files like `vision_transformer.py` and `utils.py` (from the DINO repository) for the model architecture. These files must be present in the environment where client processes run, which is why they are downloaded before starting the simulation.
- Other dependencies include `torch`, `numpy`, `flwr`, `flwr-datasets`, and potentially `wandb`.
