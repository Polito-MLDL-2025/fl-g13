"""pytorch-example: A Flower / PyTorch app."""

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl_g13.config import RAW_DATA_DIR
from fl_g13.fl_pytorch.strategy import CustomFedAvg
from fl_g13.fl_pytorch.task import (
    get_weights,
    set_weights,
)
from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import load_or_create


def gen_evaluate_fn(
        testloader: DataLoader,
        model=None,
        criterion=None,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        set_weights(model, parameters_ndarrays)
        test_loss, test_accuracy, iteration_losses = eval(testloader, model, criterion)
        return test_loss, {"centralized_accuracy": test_accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def get_data_set_default(context: Context):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar100_test = datasets.CIFAR100(root=RAW_DATA_DIR, train=False, download=True, transform=transform)
    return cifar100_test


def get_server_app(checkpoint_dir,
                   model_class,
                   optimizer=None,
                   criterion=None,
                   scheduler=None,
                   save_every=1,
                   get_datatest_fn=get_data_set_default,
                   num_rounds=10,
                   fraction_fit=1.0,  # Sample 100% of available clients for training
                   fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
                   min_fit_clients=5,  # Never sample less than 10 clients for training
                   min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
                   min_available_clients=5,  # Wait until all 10 clients are available
                   device=None,
                   use_wandb=False,
                   evaluate_fn=gen_evaluate_fn,
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
        # num_rounds = context.run_config.get("num-server-rounds") or 2
        server_device = device

        # Initialize model parameters
        ndarrays = get_weights(model)
        parameters = ndarrays_to_parameters(ndarrays)

        # Prepare dataset for central evaluation

        # This is the exact same dataset as the one donwloaded by the clients via
        # FlowerDatasets. However, we don't use FlowerDatasets for the server since
        # partitioning is not needed.
        # We make use of the "test" split only
        # global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]
        #
        # testloader = DataLoader(
        #     global_test_set.with_transform(apply_eval_transforms),
        #     batch_size=32,
        # )
        testloader = get_datatest_fn(context)
        # Define strategy
        strategy = CustomFedAvg(
            checkpoint=checkpoint_dir,
            model=model,
            run_config=context.run_config,
            use_wandb=use_wandb,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=evaluate_fn(testloader, model=model, criterion=criterion),
            evaluate_metrics_aggregation_fn=weighted_average,
            min_fit_clients=min_fit_clients,  # Never sample less than 10 clients for training
            min_evaluate_clients=min_evaluate_clients,  # Never sample less than 5 clients for evaluation
            min_available_clients=min_available_clients,
            save_every=save_every,
            start_epoch=start_epoch,
            save_best_model=save_best_model
        )
        config = ServerConfig(num_rounds=num_rounds, round_timeout=None)

        return ServerAppComponents(strategy=strategy, config=config)

    # Create ServerApp
    app = ServerApp(server_fn=server_fn)
    return app
