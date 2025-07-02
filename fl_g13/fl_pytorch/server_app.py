import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Any, Callable, Dict, Optional, Type

from fl_g13.config import RAW_DATA_DIR
from fl_g13.fl_pytorch.DynamicQuorumStrategy import DynamicQuorum
from fl_g13.fl_pytorch.datasets import get_eval_transforms
from fl_g13.fl_pytorch.strategy import CustomFedAvg
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import load_or_create


# *** -------- UTILITY FUNCTIONS FOR SERVER -------- *** #
def get_evaluate_fn(testloader, model, criterion):
    def evaluate(server_round, parameters_ndarrays, config):
        # Applies new parameters to model
        set_weights(model, parameters_ndarrays)

        # Run evaluation and return results
        test_loss, test_accuracy, _ = eval(testloader, model, criterion)
        return test_loss, {"centralized_accuracy": test_accuracy}

    return evaluate

def fit_metrics_aggregation_fn(metrics):
    losses = [n * m["train_loss"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {
        "avg_train_loss": sum(losses) / total,
    }

def evaluate_metrics_aggregation_fn(metrics):
    accuracies = [n * m["accuracy"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"decentralized_avg_eval_accuracy": sum(accuracies) / total}

def on_fit_config_fn(server_round):
    config = {
        "server_round": server_round,
    }
    return config

# *** -------- SERVER APP -------- *** #
def get_server_app(
    checkpoint_dir: str,
    prefix: str,
    model_class: Type[torch.nn.Module],
    model_config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Type[torch.optim.Optimizer]] = None,
    criterion: Optional[Type[torch.nn.Module]] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    save_every: int = 1,
    save_with_model_dir: bool = False,
    strategy: Optional[str] = None,
    get_evaluate_fn: Callable = get_evaluate_fn,
    num_rounds: int = 200,
    fraction_fit: float = 0.1,
    fraction_evaluate: float = 0.1,
    min_fit_clients: int = 10,
    min_evaluate_clients: int = 10,
    min_available_clients: int = 100,
    use_wandb: bool = False,
    wandb_config: Optional[Dict[str, Any]] = None,
    evaluate_each: int = 1,
    model: Optional[torch.nn.Module] = None,
    start_epoch: Optional[int] = None,
    global_mask: Optional[Any] = None,
    num_total_clients: int = 100,
    verbose: int = 0,
    adaptive_quorum: bool = False,
    initial_target_sparsity: float = 0.7,
    quorum_update_frequency: int = 10,
    initial_quorum: int = 1,
    quorum_increment: int = 10,
    drift_threshold: float = 0.5,
    quorum_patience: int = 2,
    force_quorum_update: int = 15,
) -> ServerApp:
    # Load a new model, if not already given
    if model is None or start_epoch is None:
        model, start_epoch = load_or_create(
            path=f"{checkpoint_dir}/{model_class.__name__}" if save_with_model_dir else checkpoint_dir,
            model_class=model_class,
            model_config=model_config,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            verbose=True,
        )

    def server_fn(context):
        if verbose > 0:
            print(f"[Server] Server on device: {next(model.parameters()).device}")

        # Retrive test dataset and prepare dataloader
        testset = datasets.CIFAR100(RAW_DATA_DIR, train=False, download=True, transform=get_eval_transforms())
        testloader = DataLoader(testset, batch_size=64)
        evaluate_fn = get_evaluate_fn(testloader, model, criterion)

        # Retrive parameters
        params = ndarrays_to_parameters(get_weights(model))

        # Call custom strategy for aggregating data
        nonlocal strategy  # Make strategy defined as param accessible under server_fn
        if strategy == 'standard' or not strategy:
            if verbose > 0:
                print("Using strategy 'CustomFedAvg' (default option)")
            strategy = CustomFedAvg(
                checkpoint_dir=checkpoint_dir,
                prefix=prefix,
                model=model,
                initial_parameters=params,
                start_epoch=start_epoch,
                save_every=save_every,
                save_with_model_dir=save_with_model_dir,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                use_wandb=use_wandb,
                wandb_config=wandb_config,
                on_fit_config_fn=on_fit_config_fn,
                evaluate_each=evaluate_each,
            )
        elif strategy == 'quorum':
            if verbose > 0:
                print("Using strategy 'AdaQuo'")
            strategy = DynamicQuorum(
                mask_sum = global_mask,
                num_total_clients = num_total_clients,
                adaptive_quorum = adaptive_quorum,
                initial_target_sparsity = initial_target_sparsity,
                quorum_update_frequency = quorum_update_frequency,
                initial_quorum = initial_quorum,
                quorum_increment = quorum_increment,
                drift_threshold = drift_threshold,
                quorum_patience = quorum_patience,
                force_quorum_update = force_quorum_update,
                
                # Default
                checkpoint_dir=checkpoint_dir,
                prefix=prefix,
                model=model,
                initial_parameters=params,
                start_epoch=start_epoch,
                save_every=save_every,
                save_with_model_dir=save_with_model_dir,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                use_wandb=use_wandb,
                wandb_config=wandb_config,
                on_fit_config_fn=on_fit_config_fn,
                evaluate_each=evaluate_each,
            )

        # Prepare server config
        rounds = context.run_config.get("num-server-rounds") or num_rounds
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
