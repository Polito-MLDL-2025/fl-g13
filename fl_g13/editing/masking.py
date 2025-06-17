import pickle
from typing import Dict

import numpy as np
import torch

from .fisher import masked_fisher_score


def _local_mask(score: Dict[str, torch.Tensor], density: float = 0.2) -> Dict[str, torch.Tensor]:
    gradient_mask = {}

    for name, scores in score.items():
        scores_flat = scores.view(-1)
        total_elements_layer = scores_flat.numel()

        # Calculate the number of elements to keep (set to 1) for this layer
        num_to_keep_layer = int(total_elements_layer * density)

        # Handle edge cases for density for this layer
        if num_to_keep_layer <= 0:
            # Keep 0 elements (mask of all zeros) for this layer
            mask = torch.zeros_like(scores, dtype=torch.float)
        elif num_to_keep_layer >= total_elements_layer:
            # Keep all elements (mask of all ones) for this layer
            mask = torch.ones_like(scores, dtype=torch.float)
        else:
            # Get the indices that would sort the flattened scores for this layer (ascending order)
            sorted_indices_layer = torch.argsort(scores_flat)

            # The indices of the elements to keep (mask = 1) are the first num_to_keep_layer indices
            indices_to_keep_layer_flat = sorted_indices_layer[:num_to_keep_layer]

            # Create a flat mask of all zeros for this layer
            flat_mask_layer = torch.zeros_like(scores_flat, dtype=torch.float)
            # Set the elements at the indices to keep to 1
            flat_mask_layer[indices_to_keep_layer_flat] = 1.0

            # Reshape the flat mask back to the original shape
            mask = flat_mask_layer.view(scores.shape)

        gradient_mask[name] = mask

    return gradient_mask


def _global_mask(score: Dict[str, torch.Tensor], density: float = 0.2) -> Dict[str, torch.Tensor]:
    gradient_mask = {}
    param_info = []  # Stores (name, shape, start_flat_idx) to reshape later

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

    # Calculate the number of elements to keep (set to 1) based on density
    num_to_keep = int(total_elements * density)

    # Handle edge cases for density (0 or 1)
    if num_to_keep <= 0:
        # If density is 0 or very close, keep 0 elements (mask of all zeros)
        gradient_mask = {name: torch.zeros(shape, dtype=torch.float, device=global_scores_flat.device) for
                         name, shape, _ in param_info}
        return gradient_mask
    elif num_to_keep >= total_elements:
        # If density is 1 or very close, keep all elements (mask of all ones)
        gradient_mask = {name: torch.ones(shape, dtype=torch.float, device=global_scores_flat.device) for name, shape, _
                         in param_info}
        return gradient_mask

    # Get the indices that would sort the flattened scores (ascending order)
    sorted_indices = torch.argsort(global_scores_flat)

    # The indices of the elements to keep (mask = 1) are the first num_to_keep indices
    # because we keep the parameters with the *smallest* scores.
    indices_to_keep_flat = sorted_indices[:num_to_keep]

    # Create a flat mask of all zeros
    flat_mask = torch.zeros(total_elements, dtype=torch.float, device=global_scores_flat.device)

    # Set the elements at the indices to keep to 1
    flat_mask[indices_to_keep_flat] = 1.0

    # Reshape the flat mask back into the dictionary structure
    current_flat_idx = 0
    for name, shape, _ in param_info:
        num_elements = shape.numel()
        layer_flat_mask = flat_mask[current_flat_idx: current_flat_idx + num_elements]
        gradient_mask[name] = layer_flat_mask.view(shape)
        current_flat_idx += num_elements

    return gradient_mask


def _create_gradiend_mask(score: Dict[str, torch.Tensor], density: float = 0.2, mask_type: str = 'local') -> Dict[
    str, torch.Tensor]:
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
) -> Dict[str, torch.Tensor] | tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # --- Parameter Validation ---
    if sparsity is not None and density is not None:
        raise ValueError("Only one of 'sparsity' or 'density' should be provided.")
    if sparsity is None and density is None:
        raise ValueError("Either 'sparsity' or 'density' must be provided.")

    if sparsity is not None:
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError(f"Sparsity value out of range, {sparsity} was given. Expected value between 0.0 and 1.0")
        target_density = 1 - sparsity
        print_param_info = f"sparsity ({sparsity:.2f})"
    else:  # density is not None
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"Density value out of range, {density} was given. Expected value between 0.0 and 1.0")
        target_density = density
        print_param_info = f"density ({density:.2f})"

    if mask_type not in ['local', 'global']:
        raise ValueError(f'Invalid mask type: {mask_type}, expected "local" or "global"')
    if rounds < 1:
        raise ValueError(f"Rounds must be a positive integer, {rounds} was given")

    # --- Initialization ---
    # Initialize mask to all ones
    mask = {name: torch.ones_like(param.data, device=param.device) for name, param in model.named_parameters()}

    # --- Calibration Rounds ---
    if rounds == 1:
        print(f'Computing simple {mask_type} mask with target {print_param_info}.')
    else:
        print(f'Computing calibrated {mask_type} mask for {rounds} rounds with target {print_param_info}.')

    for r in range(rounds):
        print(f'Round {r + 1}/{rounds}.')

        # --- Round Density ---
        current_round_density = target_density ** ((r + 1) / rounds)
        print(f'\tCurrent round density {current_round_density:.2f}%')

        # --- Compute Score ---
        print(f'\tComputing the masked fisher score')
        score = masked_fisher_score(dataloader, model, current_mask=mask)

        # --- Update Mask ---
        print(f'\tUpdating the mask')
        mask = _create_gradiend_mask(score, density=current_round_density, mask_type=mask_type)

    if return_scores:
        return mask, score

    return mask
