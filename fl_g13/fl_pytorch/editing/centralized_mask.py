from typing import List, Dict, Tuple

import torch

from fl_g13.editing import create_mask, mask_dict_to_list
from fl_g13.fl_pytorch.datasets import get_transforms, load_flwr_datasets


def get_client_masks(
        ## config client data set params
        client_partition_type='iid',  # 'iid' or 'shard' for non-iid dataset
        client_num_partitions=100,  # equal to number of client
        client_num_shards_per_partition=10,
        client_batch_size=16,
        client_dataset="cifar100",
        client_train_test_split_ratio=0.2,
        client_transform=get_transforms,
        client_seed=42,
        client_return_dataset=True,

        ## config get mask params
        mask_model=None,
        mask_sparsity=0.8,
        mask_type='global',
        mask_rounds=1,
        mask_func=None,
) -> (List[Dict[str, torch.Tensor]], List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]):
    """
    Generate pruning masks for a model based on federated client datasets.

    This function simulates a federated learning setup by partitioning a dataset into multiple client-specific
    data loaders and generating corresponding sparsity masks for each client using a specified model and
    pruning strategy.

    Parameters:
    ----------
    client_partition_type : str, optional
        Type of data partitioning: 'iid' or 'shard' (non-iid). Default is 'iid'.

    client_num_partitions : int, optional
        Total number of clients (data partitions). Default is 100.

    client_num_shards_per_partition : int, optional
        Number of shards per client, used only when partitioning is 'shard'. Default is 10.

    client_batch_size : int, optional
        Batch size used for training and validation data loaders. Default is 16.

    client_dataset : str, optional
        Dataset name to be loaded (e.g., 'cifar100'). Default is 'cifar100'.

    client_train_test_split_ratio : float, optional
        Fraction of the client dataset to use for validation. Default is 0.2.

    client_transform : callable, optional
        Transform function applied to the dataset. Should return torchvision transforms.

    client_seed : int, optional
        Random seed for dataset partitioning and reproducibility. Default is 42.

    client_return_dataset : bool, optional
        If True, returns the client train and validation dataloaders. Default is True.

    mask_model : torch.nn.Module
        The model on which sparsity masks are to be generated. Must be provided.

    mask_sparsity : float, optional
        Target sparsity level (0.0 to 1.0), where 1.0 means all weights are zeroed out. Default is 0.8.

    mask_type : str, optional
        Type of pruning: 'global' or 'local'. Global prunes across all layers; local prunes per-layer. Default is 'global'.

    mask_rounds : int, optional
        Number of iterations for the mask generation process. Default is 1.

    mask_func : callable, optional
        Optional custom mask generation function. If None, a default `create_mask` function will be used.

    Returns:
    -------
    masks : List[Dict[str, torch.Tensor]]
        A list of pruning masks (one per client), each being a dictionary mapping parameter names to binary masks.

    client_datasets : List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
        A list of tuples containing (train_loader, val_loader) for each client, if `client_return_dataset` is True.
        Otherwise, this list will be empty.

    Raises:
    ------
    ValueError
        If `mask_model` is not provided.

    """

    if mask_model is None:
        raise ValueError("mask_model is required!")
    masks = []
    client_datasets = []
    for i in range(client_num_partitions):
        partition_id = i
        client_trainloader, client_valloader = load_flwr_datasets(partition_id=partition_id,
                                                                  partition_type=client_partition_type,
                                                                  num_partitions=client_num_partitions,
                                                                  num_shards_per_partition=client_num_shards_per_partition,
                                                                  batch_size=client_batch_size,
                                                                  train_test_split_ratio=client_train_test_split_ratio,
                                                                  dataset=client_dataset,
                                                                  transform=client_transform,
                                                                  seed=client_seed
                                                                  )

        if mask_func and callable(mask_func):
            mask = mask_func(model=mask_model,
                             dataloader=client_trainloader,
                             sparsity=mask_sparsity,
                             mask_type=mask_type,
                             rounds=mask_rounds
                             )
        else:
            mask = create_mask(
                model=mask_model,
                dataloader=client_trainloader,
                sparsity=mask_sparsity,
                mask_type=mask_type,
                rounds=mask_rounds
            )
        masks.append(mask)
        if client_return_dataset:
            client_datasets.append((client_trainloader, client_valloader))
    return masks, client_datasets


def aggregate_masks(masks: List[Dict[str, torch.Tensor]],
                    strategy='union',
                    agg_func=lambda x: x[0]) \
        -> Dict[str, torch.Tensor]:
    """
    Aggregate a list of binary pruning masks using a specified strategy.

    This function combines individual client pruning masks into a single global mask.
    It supports standard aggregation strategies such as union (logical OR),
    intersection (logical AND), or a custom user-defined function.

    Parameters
    ----------
    masks : List[Dict[str, torch.Tensor]]
        A list of pruning masks, where each mask is a dictionary mapping parameter names
        to binary `torch.Tensor` masks.

    strategy : str, optional
        Aggregation strategy to use. One of:
            - 'union': combines masks using element-wise logical OR.
            - 'intersection': combines masks using element-wise logical AND.
            - 'custom': applies a user-defined aggregation function.
        Default is 'union'.

    agg_func : callable, optional
        A custom function for aggregation, used only when `strategy='custom'`.
        It must take a list of masks as input and return a single aggregated mask
        in the same format. Default is `lambda x: x[0]` (returns the first mask).

    Returns
    -------
    agg_mask : Dict[str, torch.Tensor]
        A dictionary representing the aggregated mask, mapping parameter names to
        binary tensors.

    Raises
    ------
    ValueError
        If the strategy is invalid or if `agg_func` is not callable when using 'custom'.

    Notes
    -----
    - The 'union' strategy ensures any parameter selected by at least one client is retained.
    - The 'intersection' strategy retains only parameters selected by all clients.
    - Use 'custom' for advanced or statistical aggregation (e.g., threshold-based voting).
    - Assumes all input masks have the same keys and tensor shapes.
    """
    agg_mask: Dict[str, torch.Tensor] = {}

    if not masks:
        return agg_mask  # return empty if no masks

    keys = masks[0].keys()
    dtype = next(iter(masks[0].values())).dtype  # assume all masks use same dtype

    if strategy == 'union':
        for key in keys:
            agg_bool = torch.zeros_like(masks[0][key], dtype=torch.bool)
            for mask in masks:
                agg_bool |= mask[key].bool()
            agg_mask[key] = agg_bool.to(dtype)

    elif strategy == 'intersection':
        for key in keys:
            agg_bool = torch.ones_like(masks[0][key], dtype=torch.bool)
            for mask in masks:
                agg_bool &= mask[key].bool()
            agg_mask[key] = agg_bool.to(dtype)

    elif strategy == 'custom':
        if not callable(agg_func):
            raise ValueError("agg_func must be callable!")
        agg_mask = agg_func(masks)

    else:
        raise ValueError("Invalid strategy! Use 'union', 'intersection', or 'custom'.")

    return agg_mask


def get_centralized_mask(
        ## config client data set params
        client_partition_type='iid',  # 'iid' or 'shard' for non-iid dataset
        client_num_partitions=100,  # equal to number of client
        client_num_shards_per_partition=10,
        client_batch_size=16,
        client_dataset="cifar100",
        client_train_test_split_ratio=0.2,
        client_transform=get_transforms,
        client_seed=42,
        client_return_dataset=False,

        ## config get mask params
        mask_model=None,
        mask_sparsity=0.8,
        mask_type='global',
        mask_rounds=1,
        mask_func=None,

        ## aggregate
        agg_strategy='union',
        agg_func=None
) -> List[torch.Tensor]:
    """
    Generate a centralized mask by aggregating sparsity masks computed across federated clients.

    This function simulates multiple clients using a federated data partitioning strategy, generates
    individual sparsity masks per client, and aggregates them into a single global mask using a
    specified aggregation strategy.

    Parameters
    ----------
    client_partition_type : str, optional
        Data partitioning strategy: 'iid' or 'shard' (for non-iid splits). Default is 'iid'.

    client_num_partitions : int, optional
        Number of simulated clients (i.e., partitions). Default is 100.

    client_num_shards_per_partition : int, optional
        Number of shards per partition (used for 'shard' partitioning). Default is 10.

    client_batch_size : int, optional
        Batch size for data loaders used in mask generation. Default is 16.

    client_dataset : str, optional
        Name of the dataset (e.g., 'cifar100'). Default is 'cifar100'.

    client_train_test_split_ratio : float, optional
        Ratio of client data used for validation. Default is 0.2.

    client_transform : callable, optional
        Function that applies preprocessing/transforms to the dataset. Should return torchvision transforms.

    client_seed : int, optional
        Random seed for reproducibility in dataset partitioning. Default is 42.

    client_return_dataset : bool, optional
        Whether to return client datasets. This is ignored here but passed to `get_client_masks`. Default is False.

    mask_model : torch.nn.Module
        Model on which sparsity masks are to be generated. Must be provided.

    mask_sparsity : float, optional
        Target sparsity level for the pruning masks (0.0 to 1.0). Default is 0.8.

    mask_type : str, optional
        Type of sparsity to apply: 'global' or 'local'. Default is 'global'.

    mask_rounds : int, optional
        Number of rounds/iterations for applying the mask. Default is 1.

    mask_func : callable, optional
        Optional custom function to generate the mask per client.

    agg_strategy : str, optional
        Aggregation strategy for combining masks across clients.
        Options: 'union', 'intersection', or 'custom'. Default is 'union'.

    agg_func : callable, optional
        Custom aggregation function to use when `strategy='custom'`. Must take a list of masks and return one.

    Returns
    -------
    agg_mask : Dict[str, torch.Tensor]
        A dictionary representing the aggregated global pruning mask, mapping model parameter names
        to binary `torch.Tensor` masks.

    Raises
    ------
    ValueError
        If `mask_model` is not provided or an invalid strategy is specified.

    Notes
    -----
    - This function is useful for obtaining a global view of model sparsity across decentralized data.
    - The final mask can be used to prune a model before centralized or federated fine-tuning.
    """
    masks, _ = get_client_masks(
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
        mask_func=mask_func
    )
    agg_mask = aggregate_masks(masks, strategy=agg_strategy, agg_func=agg_func)
    return mask_dict_to_list(mask_model, agg_mask)


def save_mask(mask: Dict[str, torch.Tensor], filepath: str = 'centralized_mask.pth'):
    """
    Save a binary mask dictionary in pth format.

    Parameters
    ----------
    mask : Dict[str, torch.Tensor]
        Dictionary mapping parameter names to binary mask tensors.
    filepath : str
        Path to save the mask file.
    """

    torch.save(mask, filepath)


def load_mask(filepath: str = 'centralized_mask.pth') -> Dict[str, torch.Tensor]:
    """
    Load a binary mask dictionary from either compressed JSON (sparse) or .pth (dense) format.

    Parameters
    ----------
    filepath : str
        Path to the saved mask file.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary mapping parameter names to binary mask tensors.
    """

    return torch.load(filepath)
