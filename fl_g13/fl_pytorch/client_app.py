from typing import Any, Callable, Optional, Tuple
from logging import INFO

import torch
from flwr.client import ClientApp
from flwr.common import logger
from flwr.common.typing import Context
from torch.utils.data import DataLoader

from fl_g13.fl_pytorch.client import CustomNumpyClient
from fl_g13.fl_pytorch.datasets import get_transforms, load_flwr_datasets
from fl_g13.fl_pytorch.DynamicQuorumClient import DynamicQuorumClient


def load_client_dataloaders(
    context: Context,
    partition_type: str,
    num_shards_per_partition: int,
    batch_size: int,
    train_test_split_ratio: float,
    transform: Callable = get_transforms,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and partition federated datasets for a client.

    Args:
        context (Context): The Flower client context.
        partition_type (str): The type of data partitioning ('iid' or 'shard').
        num_shards_per_partition (int): The number of shards per partition for non-IID.
        batch_size (int): The batch size for the DataLoaders.
        train_test_split_ratio (float): The ratio for the train-test split.
        transform (Callable): The function to apply transformations to the data.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation DataLoaders.
    """
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader = load_flwr_datasets(
        partition_id=partition_id,
        partition_type=partition_type,
        num_partitions=num_partitions,
        num_shards_per_partition=num_shards_per_partition,
        batch_size=batch_size,
        train_test_split_ratio=train_test_split_ratio,
        transform=transform,
    )
    return trainloader, valloader

def get_client_app(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    strategy: Optional[str] = None,
    load_data_fn: Callable = load_client_dataloaders,
    batch_size: int = 64,
    partition_type: str = "iid",
    num_shards_per_partition: int = 2,
    train_test_split_ratio: float = 0.2,
    local_epochs: int = 1,
    local_steps: int = 4,
    model_editing: bool = False,
    mask_type: str = "global",
    sparsity: float = 0.2,
    mask_calibration_round: int = 1,
    model_editing_batch_size: int = 1,
    is_save_weights_to_state: bool = False,
    mask: Optional[Any] = None,
    warm_up_rounds: int = 0,
    mask_func: Optional[Callable] = None,
    verbose: int = 0,
) -> ClientApp:
    """
    Create and configure a Flower ClientApp.

    This function initializes a client for federated learning, supporting different
    strategies like standard FedAvg and dynamic quorum approach.

    Args:
        model (torch.nn.Module): The PyTorch model.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (Optional[Any]): The learning rate scheduler.
        device (torch.device): The device to run on.
        strategy (Optional[str]): The client-side strategy ('standard' or 'quorum').
        load_data_fn (Callable): The function to load client data.
        batch_size (int): The batch size.
        partition_type (str): The data partition type.
        num_shards_per_partition (int): The number of shards per partition.
        train_test_split_ratio (float): The train-test split ratio.
        local_epochs (int): The number of local epochs.
        local_steps (int): The number of local steps.
        model_editing (bool): Enables model editing.
        mask_type (str): The type of mask to use.
        sparsity (float): The target mask sparsity.
        mask_calibration_round (int): Rounds for mask calibration.
        model_editing_batch_size (int): The batch size for model editing.
        is_save_weights_to_state (bool): Saves weights to client state.
        mask (Optional[Any]): An initial mask.
        warm_up_rounds (int): The number of warm-up rounds.
        mask_func (Optional[Callable]): A function to generate the mask.
        verbose (int): The verbosity level.

    Returns:
        ClientApp: The configured Flower ClientApp.
    """

    def client_fn(context: Context) -> ClientApp:
        """The client function that Flower will call to produce a client."""
        if verbose > 0:
            logger.log(INFO, f"[Client] Client on device: {next(model.parameters()).device}")

        trainloader, valloader = load_data_fn(
            context=context,
            partition_type=partition_type,
            batch_size=batch_size,
            num_shards_per_partition=num_shards_per_partition,
            train_test_split_ratio=train_test_split_ratio,
        )
        client_state = context.state

        client: Any
        if strategy == "standard" or not strategy:
            if verbose > 0:
                logger.log(INFO, f"[Client] Using default client 'CustomNumpyClient'")
            client = CustomNumpyClient(
                client_state=client_state,
                local_epochs=local_epochs,
                trainloader=trainloader,
                valloader=valloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model_editing=model_editing,
                mask_type=mask_type,
                sparsity=sparsity,
                mask_calibration_round=mask_calibration_round,
                model_editing_batch_size=model_editing_batch_size,
                mask_func=mask_func,
                mask=mask,
                local_steps=local_steps,
                verbose=verbose,
            )
        elif strategy == "quorum":
            if verbose > 0:
                logger.log(INFO, f"[Client] Using client 'DynamicQuorumClient'")
            client = DynamicQuorumClient(
                client_state=client_state,
                local_epochs=local_epochs,
                trainloader=trainloader,
                valloader=valloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model_editing=model_editing,
                mask_type=mask_type,
                sparsity=sparsity,
                mask_calibration_round=mask_calibration_round,
                model_editing_batch_size=model_editing_batch_size,
                mask_func=mask_func,
                mask=mask,
                local_steps=local_steps,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return client.to_client()

    return ClientApp(client_fn=client_fn)