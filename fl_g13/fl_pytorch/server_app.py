"""pytorch-example: A Flower / PyTorch app."""

import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from fl_g13.config import RAW_DATA_DIR
from fl_g13.fl_pytorch.strategy import SaveModelFedAvg
from fl_g13.fl_pytorch.task import (
    get_weights,
    set_weights,
)
from typing import List, Tuple
from fl_g13.fl_pytorch.datasets import get_eval_transforms
#from fl_g13.fl_pytorch.task import test
from datasets import load_dataset
from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import load_or_create
from torchvision import datasets


def get_evaluate_fn(
        testloader: DataLoader,
        model=None,
        criterion=None,
):
    """Generate the function for centralized evaluation of the global model on full test set. 
    Executed by the server at the end of each round."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        set_weights(model, parameters_ndarrays)
        test_loss, test_accuracy, iteration_losses = eval(testloader, model, criterion)
        return test_loss, {"centralized_accuracy": test_accuracy}

    return evaluate


def on_fit_config(server_round: int) -> Metrics:
    """Allow communication from server to client.
    Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate application specific metrics computed by clients at each round with .evaluate()"""
    # Multiply accuracy calculated by each client during .evaluate() by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Allow to communicate metrics from client to server.
    Aggregate application specific metrics computed by clients at each round with .fit()"""
    
    for _, m in metrics:
        print(f"client train loss: {m['train_loss']}")
        print(f"client drift: {m['drift']}")
    
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_drifts = [num_examples * m["drift"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric
    return {"avg_train_loss": sum(train_losses) / sum(examples), "avg_drift": sum(train_drifts) / sum(examples)}


def get_server_app(checkpoint_dir,
                   model_class,
                   optimizer=None,
                   criterion=None,
                   scheduler=None,
                   save_every=1,
                   num_rounds=200,
                   fraction_fit=0.1,  # Sample 10% of available clients for training
                   fraction_evaluate=0.1,  # Sample 10% of available clients for evaluation
                   min_fit_clients=10,  # Never sample less than 10 clients for training
                   min_evaluate_clients=10,  # Never sample less than 10 clients for evaluation
                   min_available_clients=100,  # Wait until all 100 clients are available
                   device=None,
                   use_wandb=False,
                   save_best_model=False
                   ):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, start_epoch = load_or_create(
        path=checkpoint_dir,
        model_class=model_class,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
    )

    def server_fn(context: Context):
        print(f'Continue train model from epoch {start_epoch}')
        # Read from config
        # run_config holds hyperparameters that we might want to override at runtime
        number_rounds = context.run_config.get("num-server-rounds") or num_rounds #defined in .toml

        # Initialize model parameters
        ndarrays = get_weights(model)
        parameters = ndarrays_to_parameters(ndarrays)

        #testset = load_dataset("cifar100", split="test")
        testset = datasets.CIFAR100(RAW_DATA_DIR, train=False, download=True, transform=get_eval_transforms())

        # load global full testset for central evaluation
        testloader = DataLoader(testset, batch_size=32)
        
        # Define strategy
        strategy = SaveModelFedAvg(
            checkpoint=checkpoint_dir,
            model=model,
            run_config=context.run_config,
            use_wandb=use_wandb,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=get_evaluate_fn(testloader, model, criterion), 
            evaluate_metrics_aggregation_fn=weighted_average,
            min_fit_clients=min_fit_clients, 
            min_evaluate_clients=min_evaluate_clients, 
            min_available_clients=min_available_clients,
            save_every=save_every,
            start_epoch =start_epoch,
            fit_metrics_aggregation_fn=handle_fit_metrics,
            save_best_model=save_best_model,
        )
        config = ServerConfig(num_rounds=number_rounds, round_timeout=None)

        return ServerAppComponents(strategy=strategy, config=config)

    # Create ServerApp
    app = ServerApp(server_fn=server_fn)
    return app