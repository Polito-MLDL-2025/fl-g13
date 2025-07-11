import pickle
from typing import Dict

import numpy as np
import torch

from .fisher import masked_fisher_score

def _local_mask(score: Dict[str, torch.Tensor], density: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    Creates a binary mask for each parameter layer locally based on a density threshold.

    For each layer, the scores are sorted, and a mask is created to keep the
    `density` proportion of the scores with the smallest values.

    Args:
        score (Dict[str, torch.Tensor]): A dictionary mapping parameter names to score tensors.
        density (float, optional): The fraction of weights to keep for each layer. 
            Defaults to 0.2.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the binary masks for each layer.
    """
    gradient_mask = {}

    for name, scores in score.items():
        scores_flat = scores.view(-1)
        
        total_elements_layer = scores_flat.numel()
        num_to_keep_layer = int(total_elements_layer * density)

        # Handle edge cases for density for this layer
        if num_to_keep_layer <= 0:
            mask = torch.zeros_like(scores, dtype=torch.float) # Zero mask
        elif num_to_keep_layer >= total_elements_layer:
            mask = torch.ones_like(scores, dtype=torch.float) # One mask
        else:
            # Get the indices that would sort the flattened scores for this layer (ascending order)
            sorted_indices_layer = torch.argsort(scores_flat)

            # The indices of the elements to keep (mask = 1) are the first num_to_keep_layer indices
            indices_to_keep_layer_flat = sorted_indices_layer[:num_to_keep_layer]

            flat_mask_layer = torch.zeros_like(scores_flat, dtype=torch.float)
            flat_mask_layer[indices_to_keep_layer_flat] = 1.0

            # Reshape the mask
            mask = flat_mask_layer.view(scores.shape)

        gradient_mask[name] = mask

    return gradient_mask

def _global_mask(score: Dict[str, torch.Tensor], density: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    Creates a binary mask for all parameters globally based on a density threshold.

    All scores from all layers are flattened and concatenated. A single threshold
    is then applied to keep the `density` proportion of the scores with the
    smallest values across the entire model.

    Args:
        score (Dict[str, torch.Tensor]): A dictionary mapping parameter names to score tensors.
        density (float, optional): The fraction of weights to keep for the entire model. 
            Defaults to 0.2.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the binary masks for each layer.
    """
    gradient_mask = {}
    param_info = []

    # Flatten scores and store info for reshaping
    all_scores_flat_list = []
    current_flat_idx = 0
    for name, scores in score.items():
        shape = scores.shape
        num_elements = scores.numel()
        param_info.append((name, shape, current_flat_idx))
        all_scores_flat_list.append(scores.flatten())
        current_flat_idx += num_elements

    global_scores_flat = torch.cat(all_scores_flat_list)
    
    total_elements = global_scores_flat.numel()
    num_to_keep = int(total_elements * density)

    # Handle edge cases for density (0 or 1)
    if num_to_keep <= 0:
        gradient_mask = {
            name: torch.zeros(shape, dtype=torch.float, device=global_scores_flat.device) 
                for name, shape, _ in param_info
        }
        return gradient_mask
    elif num_to_keep >= total_elements:
        gradient_mask = {
            name: torch.ones(shape, dtype=torch.float, device=global_scores_flat.device) 
                for name, shape, _ in param_info
        }
        return gradient_mask

    # Get the indices that would sort the flattened scores (ascending order)
    sorted_indices = torch.argsort(global_scores_flat)

    # The indices of the elements to keep (mask = 1) are the first num_to_keep_layer indices
    indices_to_keep_flat = sorted_indices[:num_to_keep]

    flat_mask = torch.zeros(total_elements, dtype=torch.float, device=global_scores_flat.device)
    flat_mask[indices_to_keep_flat] = 1.0

    # Reshape the flat mask back into the dictionary structure
    current_flat_idx = 0
    for name, shape, _ in param_info:
        num_elements = shape.numel()
        layer_flat_mask = flat_mask[current_flat_idx: current_flat_idx + num_elements]
        gradient_mask[name] = layer_flat_mask.view(shape)
        current_flat_idx += num_elements

    return gradient_mask

def _validate_and_get_density(sparsity: float | None, density: float | None) -> float:
    """
    Validates sparsity and density parameters and returns the target density.

    Args:
        sparsity (float | None): The fraction of weights to prune (0.0 to 1.0).
        density (float | None): The fraction of weights to keep (0.0 to 1.0).

    Returns:
        float: The target density value.

    Raises:
        ValueError: If both or neither sparsity and density are provided, or if
                    their values are out of the [0, 1] range.
    """
    if sparsity is not None and density is not None:
        raise ValueError("Only one of 'sparsity' or 'density' should be provided.")
    if sparsity is None and density is None:
        raise ValueError("Either 'sparsity' or 'density' must be provided.")

    if sparsity is not None:
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError(f"Sparsity value out of range, {sparsity} was given. Expected value between 0.0 and 1.0")
        return 1 - sparsity
    else:  # density is not None
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"Density value out of range, {density} was given. Expected value between 0.0 and 1.0")
        return density

def create_mask_from_scores(
    score: Dict[str, torch.Tensor], 
    sparsity: float | None = None, 
    density: float | None = None, 
    mask_type: str = 'local'
) -> Dict[str, torch.Tensor]:
    """
    Creates a binary mask from scores based on either sparsity or density.

    This function serves as a wrapper around the local and global mask creation
    functions, providing a single interface for mask generation.

    Args:
        score (Dict[str, torch.Tensor]): A dictionary of parameter names to score tensors.
        sparsity (float | None, optional): The target sparsity level. Defaults to None.
        density (float | None, optional): The target density level. Defaults to None.
        mask_type (str, optional): The type of mask to create ('local' or 'global').
            Defaults to 'local'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the generated binary masks.
    """
    # --- Parameter Validation ---
    target_density = _validate_and_get_density(sparsity, density)

    if mask_type not in ['local', 'global']:
        raise ValueError(f'Invalid mask type: {mask_type}, expected "local" or "global"')
    
    return _create_gradient_mask(
        score=score,
        density = target_density,
        mask_type = mask_type
    )

def _create_gradient_mask(score: Dict[str, torch.Tensor], density: float = 0.2, mask_type: str = 'local') -> Dict[
    str, torch.Tensor]:
    """
    Dispatches to the appropriate mask creation function based on mask_type.

    Args:
        score (Dict[str, torch.Tensor]): A dictionary of parameter names to score tensors.
        density (float, optional): The target density level. Defaults to 0.2.
        mask_type (str, optional): The type of mask to create ('local' or 'global'). 
            Defaults to 'local'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the generated binary masks.

    Raises:
        ValueError: If an invalid mask_type is provided.
    """
    if mask_type == 'local':
        gradient_mask = _local_mask(score, density)
    elif mask_type == 'global':
        gradient_mask = _global_mask(score, density)
    else:
        raise ValueError(f'Invalid mask type: {mask_type}, expected "local" or "global".')

    return gradient_mask

def mask_dict_to_list(model, gradient_mask_dict):
    """
    Converts a {param_name: mask_tensor} dict into a list of mask_tensors
    in the same order as model.parameters().
    """
    mask_list = []
    for name, param in model.named_parameters():
        if name not in gradient_mask_dict:
            mask = torch.zeros_like(param, dtype=torch.float)
        else:
            mask = gradient_mask_dict[name]

        if param.shape != mask.shape:
            raise ValueError(f"Mask shape mismatch for {name}: {param.shape} vs {mask.shape}")
        mask_list.append(mask.to(param.device))
    return mask_list

def compress_mask_sparse(mask_list):
    """
    Compresses a list of sparse binary masks into a byte string.

    Args:
        mask_list (list[torch.Tensor]): A list of binary mask tensors.

    Returns:
        bytes: The serialized representation of the compressed masks.
    """
    compressed = []
    for mask in mask_list:
        flat_mask = mask.view(-1)
        indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1).tolist()
        shape = list(mask.shape)
        compressed.append((shape, indices))
    data_bytes = pickle.dumps(compressed)
    return data_bytes

def uncompress_mask_sparse(mask_bytes, device=None):
    """
    Decompresses a byte string into a list of sparse binary masks.

    Args:
        mask_bytes (bytes): The serialized representation of the compressed masks.
        device (torch.device, optional): The device to place the uncompressed tensors on. 
            Defaults to None.

    Returns:
        list[torch.Tensor]: A list of the reconstructed binary mask tensors.
    """
    compressed_list = pickle.loads(mask_bytes)
    uncompressed = []
    for shape, indices in compressed_list:
        flat = torch.zeros(int(np.prod(shape)), dtype=torch.float32)
        flat[indices] = 1.0
        mask = flat.view(shape)
        if device:
            mask = mask.to(device)
        uncompressed.append(mask)
    return uncompressed

def create_mask(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        sparsity: float | None = None,
        density: float | None = None,
        mask_type: str = 'local',
        rounds: int = 1,
        return_scores: bool = False,
        verbose: bool = False
) -> Dict[str, torch.Tensor] | tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Computes a binary mask for a model using iterative pruning.

    This function can perform one-shot pruning or iterative pruning over multiple
    rounds. In each round, it computes Fisher scores, creates a mask, and then
    applies it before the next round.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for computing scores.
        model (torch.nn.Module): The model to be pruned.
        sparsity (float | None, optional): The target sparsity level. Defaults to None.
        density (float | None, optional): The target density level. Defaults to None.
        mask_type (str, optional): The type of mask to create ('local' or 'global'). 
            Defaults to 'local'.
        rounds (int, optional): The number of pruning rounds. Defaults to 1.
        return_scores (bool, optional): Whether to return the final scores along with the mask. 
            Defaults to False.
        verbose (bool, optional): Whether to print progress information. Defaults to False.

    Returns:
        Dict[str, torch.Tensor] | tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: 
            The final binary mask, and optionally the computed scores.
    """
    # --- Parameter Validation ---
    target_density = _validate_and_get_density(sparsity, density)
    print_param_info = f"sparsity ({1 - target_density:.2f})" if sparsity is not None else f"density ({target_density:.2f})"

    if mask_type not in ['local', 'global']:
        raise ValueError(f'Invalid mask type: {mask_type}, expected "local" or "global"')
    if rounds < 1:
        raise ValueError(f"Rounds must be a positive integer, {rounds} was given")

    # --- Initialization ---
    # Initialize mask to all ones
    mask = {name: torch.ones_like(param.data, device=param.device) for name, param in model.named_parameters()}

    # --- Calibration Rounds ---
    if rounds == 1:
        if verbose:
            print(f'Computing simple {mask_type} mask with target {print_param_info}.')
    else:
        if verbose:
            print(f'Computing calibrated {mask_type} mask for {rounds} rounds with target {print_param_info}.')

    for r in range(rounds):
        if verbose:
            print(f'Round {r + 1}/{rounds}.')

        # --- Round Density ---
        current_round_density = target_density ** ((r + 1) / rounds)
        if verbose:
            print(f'\tCurrent round density {100*current_round_density:.2f}%')

        # --- Compute Score ---
        if verbose:
            print(f'\tComputing the masked fisher score')
        score = masked_fisher_score(dataloader, model, current_mask=mask, verbose = verbose)

        # --- Update Mask ---
        if verbose:
            print(f'\tUpdating the mask')
        mask = _create_gradient_mask(score, density=current_round_density, mask_type=mask_type)

    if return_scores:
        return mask, score

    return mask

def compute_mask_stats(mask_dict):
    """
    Computes statistics about a mask, both overall and on a per-layer basis.

    Calculates the number of total, kept (1s), and masked (0s) elements, as
    well as the density and sparsity of the mask.

    Args:
        mask_dict (Dict[str, torch.Tensor]): A dictionary of layer names to mask tensors.

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    stats = {}

    # --- Overall Statistics ---
    total_elements = 0
    kept_elements_overall = 0 # Elements with value 1
    masked_elements_overall = 0 # Elements with value 0

    for name, mask_tensor in mask_dict.items():
        num_elements = mask_tensor.numel()
        kept_in_layer = torch.sum(mask_tensor == 1).item()
        masked_in_layer = num_elements - kept_in_layer

        total_elements += num_elements
        kept_elements_overall += kept_in_layer
        masked_elements_overall += masked_in_layer

        # --- Layer-wise Statistics ---
        layer_stats = {
            'num_elements': num_elements,
            'kept_elements': kept_in_layer,
            'masked_elements': masked_in_layer,
            'density': kept_in_layer / num_elements if num_elements > 0 else 0.0,
            'sparsity': masked_in_layer / num_elements if num_elements > 0 else 0.0
        }
        stats[name] = layer_stats

    # --- Add Overall Statistics to the result dictionary ---
    stats['overall'] = {
        'total_elements': total_elements,
        'kept_elements': kept_elements_overall,
        'masked_elements': masked_elements_overall,
        'density': kept_elements_overall / total_elements if total_elements > 0 else 0.0,
        'sparsity': masked_elements_overall / total_elements if total_elements > 0 else 0.0
    }

    return stats

def format_mask_stats(stats, layer = False) -> str:
    """
    Formats mask statistics into a human-readable string.

    Args:
        stats (dict): A dictionary of mask statistics, as returned by `compute_mask_stats`.
        layer (bool, optional): Whether to include layer-wise statistics in the output. 
            Defaults to False.

    Returns:
        str: A formatted string containing the mask statistics.
    """
    if 'overall' not in stats:
        return "Invalid stats dictionary format."

    output_lines = []
    overall_stats = stats['overall']
    output_lines.append("--- Overall Mask Statistics ---")
    output_lines.append(f"Total Elements: {overall_stats['total_elements']}")
    output_lines.append(f"Kept Elements (1s): {overall_stats['kept_elements']}")
    output_lines.append(f"Masked Elements (0s): {overall_stats['masked_elements']}")
    output_lines.append(f"Overall Density: {overall_stats['density']:.4f}")
    output_lines.append(f"Overall Sparsity: {overall_stats['sparsity']:.4f}")
    output_lines.append("-" * 30)

    if not layer:
        return "\n".join(output_lines)

    output_lines.append("")
    output_lines.append("--- Layer-wise Mask Statistics ---")
    # Sort layer names for consistent output
    layer_names = sorted([name for name in stats if name != 'overall'])
    for name in layer_names:
        layer_stats = stats[name]
        output_lines.append(f"Layer: {name}")
        output_lines.append(f"  Num Elements: {layer_stats['num_elements']}")
        output_lines.append(f"  Kept Elements: {layer_stats['kept_elements']}")
        output_lines.append(f"  Masked Elements: {layer_stats['masked_elements']}")
        output_lines.append(f"  Density: {layer_stats['density']:.4f}")
        output_lines.append(f"  Sparsity: {layer_stats['sparsity']:.4f}")
        output_lines.append("-" * 20)
    
    return "\n".join(output_lines)