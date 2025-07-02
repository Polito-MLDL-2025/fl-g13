import gc
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fl_g13.editing import create_mask, mask_dict_to_list
from fl_g13.fl_pytorch.datasets import get_transforms, load_flwr_datasets


def get_client_masks(
    # Client configuration
    client_partition_type: str = "iid",
    client_num_partitions: int = 100,
    client_num_shards_per_partition: int = 10,
    client_batch_size: int = 1,
    client_dataset: str = "cifar100",
    client_train_test_split_ratio: float = 0.2,
    client_transform: Callable = get_transforms,
    client_seed: int = 42,
    client_return_dataset: bool = False,
    # Mask generation parameters
    mask_model: Optional[torch.nn.Module] = None,
    mask_sparsity: float = 0.8,
    mask_type: str = "global",
    mask_rounds: int = 1,
    mask_func: Optional[Callable] = None,
    mask_store_in_cpu: bool = True,
    # Fisher score parameters
    return_scores: bool = True,
) -> Tuple[
    List[Dict[str, torch.Tensor]],  # Masks
    List[Dict[str, torch.Tensor]],  # Scores
    List[Tuple[DataLoader, DataLoader]],  # Datasets
]:
    """
    Generate masks and (optionally) scores for a specified number of clients.

    This function simulates a federated environment where each client computes a
    mask for a given model based on its local data. It can also return the
    raw scores used for mask generation and the client datasets.

    Args:
        client_partition_type (str): The data partitioning strategy ('iid' or 'shard').
        client_num_partitions (int): The total number of clients.
        client_num_shards_per_partition (int): Shards per partition for non-IID data.
        client_batch_size (int): The batch size for client DataLoaders.
        client_dataset (str): The dataset to use (e.g., 'cifar100').
        client_train_test_split_ratio (float): The train-test split ratio for client data.
        client_transform (Callable): The function for data transformations.
        client_seed (int): The random seed for data partitioning.
        client_return_dataset (bool): If True, returns the client DataLoaders.
        mask_model (torch.nn.Module): The model to generate masks for.
        mask_sparsity (float): The target sparsity for the masks.
        mask_type (str): The type of mask to generate ('global' or 'local').
        mask_rounds (int): The number of rounds for mask calibration.
        mask_func (Optional[Callable]): A custom function for mask generation.
        mask_store_in_cpu (bool): If True, stores masks and scores on the CPU.
        return_scores (bool): If True, returns the scores alongside the masks.

    Returns:
        Tuple: A tuple containing lists of masks, scores, and client datasets.
    """
    if mask_model is None:
        raise ValueError("A `mask_model` must be provided.")

    masks: List[Dict[str, torch.Tensor]] = []
    scores: List[Dict[str, torch.Tensor]] = []
    client_datasets: List[Tuple[DataLoader, DataLoader]] = []

    for i in tqdm(range(client_num_partitions), desc="Generating client masks"):
        # Load the dataset for the current client
        client_trainloader, client_valloader = load_flwr_datasets(
            partition_id=i,
            partition_type=client_partition_type,
            num_partitions=client_num_partitions,
            num_shards_per_partition=client_num_shards_per_partition,
            batch_size=client_batch_size,
            train_test_split_ratio=client_train_test_split_ratio,
            dataset=client_dataset,
            transform=client_transform,
            seed=client_seed,
        )

        # Use the default `create_mask` function if none is provided
        if not (mask_func and callable(mask_func)):
            mask_func = create_mask

        # Generate the mask (and scores if requested)
        mask_result = mask_func(
            model=mask_model,
            dataloader=client_trainloader,
            sparsity=mask_sparsity,
            mask_type=mask_type,
            rounds=mask_rounds,
            return_scores=return_scores,
        )

        if return_scores:
            mask, score = mask_result
            if mask_store_in_cpu:
                score = {key: tensor.cpu() for key, tensor in score.items()}
            scores.append(score)
        else:
            mask = mask_result

        if mask_store_in_cpu:
            mask = {key: tensor.cpu() for key, tensor in mask.items()}
        masks.append(mask)

        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        if client_return_dataset:
            client_datasets.append((client_trainloader, client_valloader))

    return masks, scores, client_datasets

def aggregate_by_sum(
    masks: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Aggregate a list of masks by summing them element-wise.

    Args:
        masks (List[Dict[str, torch.Tensor]]): A list of masks to aggregate.

    Returns:
        Dict[str, torch.Tensor]: The aggregated mask.
    """
    if not masks:
        return {}

    # Initialize the aggregated mask with zeros
    aggregated = {k: torch.zeros_like(v) for k, v in masks[0].items()}

    # Sum the masks
    for mask in masks:
        for k in mask:
            aggregated[k] += mask[k]

    return aggregated

def aggregate_masks(
    masks: List[Dict[str, torch.Tensor]],
    strategy: str = "union",
    agg_func: Optional[Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate a list of masks using a specified strategy.

    Args:
        masks (List[Dict[str, torch.Tensor]]): The list of masks to aggregate.
        strategy (str): The aggregation strategy ('union', 'intersection', or 'custom').
        agg_func (Optional[Callable]): A custom aggregation function for the 'custom' strategy.

    Returns:
        Dict[str, torch.Tensor]: The aggregated mask.
    """
    if not masks:
        return {}

    keys = masks[0].keys()
    dtype = next(iter(masks[0].values())).dtype
    agg_mask: Dict[str, torch.Tensor] = {}

    if strategy == "union":
        for key in keys:
            agg_bool = torch.zeros_like(masks[0][key], dtype=torch.bool)
            for mask in masks:
                agg_bool |= mask[key].bool()
            agg_mask[key] = agg_bool.to(dtype)
    elif strategy == "intersection":
        for key in keys:
            agg_bool = torch.ones_like(masks[0][key], dtype=torch.bool)
            for mask in masks:
                agg_bool &= mask[key].bool()
            agg_mask[key] = agg_bool.to(dtype)
    elif strategy == "custom":
        if not callable(agg_func):
            raise ValueError("`agg_func` must be a callable for the 'custom' strategy.")
        agg_mask = agg_func(masks)
    else:
        raise ValueError("Invalid strategy. Choose from 'union', 'intersection', or 'custom'.")

    return agg_mask

def get_centralized_mask(
    # Client data configuration
    client_partition_type: str = "iid",
    client_num_partitions: int = 100,
    client_num_shards_per_partition: int = 10,
    client_batch_size: int = 16,
    client_dataset: str = "cifar100",
    client_train_test_split_ratio: float = 0.2,
    client_transform: Callable = get_transforms,
    client_seed: int = 42,
    client_return_dataset: bool = False,
    # Mask generation parameters
    mask_model: Optional[torch.nn.Module] = None,
    mask_sparsity: float = 0.8,
    mask_type: str = "global",
    mask_rounds: int = 1,
    mask_func: Optional[Callable] = None,
    # Aggregation parameters
    agg_strategy: str = "union",
    agg_func: Optional[Callable] = None,
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generate and aggregate client masks to create a centralized mask.

    Args:
        client_partition_type (str): The data partitioning strategy.
        client_num_partitions (int): The total number of clients.
        client_num_shards_per_partition (int): Shards per partition for non-IID.
        client_batch_size (int): The batch size for client DataLoaders.
        client_dataset (str): The dataset to use.
        client_train_test_split_ratio (float): The train-test split ratio.
        client_transform (Callable): The function for data transformations.
        client_seed (int): The random seed for data partitioning.
        client_return_dataset (bool): If True, returns client datasets.
        mask_model (torch.nn.Module): The model to generate masks for.
        mask_sparsity (float): The target sparsity for the masks.
        mask_type (str): The type of mask to generate.
        mask_rounds (int): The number of rounds for mask calibration.
        mask_func (Optional[Callable]): A custom function for mask generation.
        agg_strategy (str): The aggregation strategy.
        agg_func (Optional[Callable]): A custom aggregation function.

    Returns:
        Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]: The centralized mask as a list and a dictionary.
    """
    masks, _, _ = get_client_masks(
        client_partition_type=client_partition_type,
        client_num_partitions=client_num_partitions,
        client_num_shards_per_partition=client_num_shards_per_partition,
        client_batch_size=client_batch_size,
        client_dataset=client_dataset,
        client_train_test_split_ratio=client_train_test_split_ratio,
        client_transform=client_transform,
        client_seed=client_seed,
        client_return_dataset=client_return_dataset,
        mask_model=mask_model,
        mask_sparsity=mask_sparsity,
        mask_type=mask_type,
        mask_rounds=mask_rounds,
        mask_func=mask_func,
    )
    agg_mask = aggregate_masks(masks, strategy=agg_strategy, agg_func=agg_func)
    return mask_dict_to_list(mask_model, agg_mask), agg_mask

def save_mask(mask: Dict[str, torch.Tensor], filepath: str = "centralized_mask.pth") -> None:
    """Save a mask dictionary to a file."""
    torch.save(mask, filepath)

def save_masks_scores(
    masks: List[Dict[str, torch.Tensor]],
    scores: List[Dict[str, torch.Tensor]],
    filepath: str = "client_masks.pth",
) -> None:
    """Save lists of masks and scores to a file."""
    data = (masks, scores)
    torch.save(data, filepath)

def load_masks_scores(
    filepath: str = "centralized_masks.pth",
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """Load lists of masks and scores from a file."""
    masks, scores = torch.load(filepath)
    return masks, scores

def load_mask(filepath: str = "centralized_mask.pth") -> Dict[str, torch.Tensor]:
    """Load a mask dictionary from a file."""
    return torch.load(filepath)

def compute_masks_and_scores(
    model: torch.nn.Module,
    client_partition_type: str,
    client_num_shards_per_partition: int,
    sparsity: float = 0.7,
    mask_type: str = "global",
    mask_rounds: int = 3,
    num_clients: int = 100,
    client_batch_size: int = 1,
    file_name: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Compute and save client masks and scores, or load them if they already exist.

    Args:
        model (torch.nn.Module): The model to generate masks for.
        client_partition_type (str): The data partitioning strategy.
        client_num_shards_per_partition (int): Shards per partition for non-IID.
        sparsity (float): The target sparsity for the masks.
        mask_type (str): The type of mask to generate.
        mask_rounds (int): The number of rounds for mask calibration.
        num_clients (int): The total number of clients.
        client_batch_size (int): The batch size for client DataLoaders.
        file_name (Optional[str]): The file to save or load the masks and scores from.
        verbose (bool): Inf True, prints progress informatio.

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]: The client masks and scores.
    """
    if file_name and os.path.isfile(file_name):
        if verbose:
            print(f"[CENTR_MASK] Found {file_name}. Loading masks and scores from memory.")
        return load_masks_scores(file_name)

    if verbose:
        print("[CENTR_MASK] Computing masks and scores.")

    client_masks, scores, _ = get_client_masks(
        client_partition_type=client_partition_type,
        client_num_partitions=num_clients,
        client_num_shards_per_partition=client_num_shards_per_partition,
        client_batch_size=client_batch_size,
        mask_model=model,
        mask_sparsity=sparsity,
        mask_type=mask_type,
        mask_rounds=mask_rounds,
        return_scores=True,  # Always return scores
    )

    if not file_name:
        file_name = "centralized_mask.pth"

    if verbose:
        print(f'[CENTR_MASK] Saving masks and scores to "{file_name}"')
    save_masks_scores(client_masks, scores, file_name)

    return client_masks, scores