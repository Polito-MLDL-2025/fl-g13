import pickle

import numpy as np
import torch

from fisher import masked_fisher_score

def _local_mask(class_score, sparsity = 0.2):
    """Create a mask for each parameter

    Args:
        class_score (_type_): scores related for a specific class
        sparsity (float, optional): expected sparsity of each parameter's mask. Defaults to 0.2.

    Returns:
        dict: dict of {param_name: binary_mask_tensor}
    """
    gradient_mask = {}

    for name, scores in class_score.items():
        scores_flat = scores.view(-1)
        k = int(len(scores_flat) * sparsity)

        if k == 0:
            # Prevent empty mask
            mask = torch.zeros_like(scores, dtype = torch.float)
        else:
            threshold, _ = torch.kthvalue(scores_flat, k)
            mask = (scores <= threshold).to(dtype = torch.float)

        gradient_mask[name] = mask

    return gradient_mask

def _global_mask(class_score, sparsity = 0.2):
    gradient_mask = {}

    global_scores = torch.cat([torch.flatten(v) for v in class_score.values()])
    # Calculate the number of parameters to keep based on the desired sparsity
    k = int(global_scores.numel() * sparsity)
    # Proceed only if there is at least one parameter to keep
    if k >= 1:
        # Find the k-th smallest value in the global scores (this is the threshold)
        threshold, _ = torch.kthvalue(global_scores, k)
        
        for name, scores in class_score.items():
            mask = (scores <= threshold).to(dtype = torch.float)
            gradient_mask[name] = mask

    return gradient_mask

def create_gradiend_mask(class_score, sparsity = 0.2, mask_type = 'local'):
    """
    class_score: dict of {param_name: tensor}
    sparsity: fraction of parameters to keep editable (lowest scores values)

    Returns: dict {param_name: binary_mask_tensor}
    """
    if mask_type == 'local':
        gradient_mask = _local_mask(class_score, sparsity)
    elif mask_type == 'global':
        gradient_mask = _global_mask(class_score, sparsity)
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
            mask = torch.zeros_like(param, dtype = torch.float)
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

def create_calibrated_mask(dataloader, model, sparsity = 0.2, mask_type = 'local', rounds = 5):
    # Initialize mask (full 1)
    mask = [torch.ones_like(p, device = p.device) for p in model.parameters()]
    # Invert sparsity. Compute which parameters needs to be set to 0
    sparsity = 1 - sparsity
    
    print(f'Computing calibrated mask for {rounds} rounds.')
    for r in range(rounds):
        print(f'Round {r + 1}.')
        # Target sparsity
        s = sparsity**((r + 1)/rounds)
        print(f'\tTarget sparsity {s:.2f}')
        
        # Compute score
        print(f'\tComputing the masked fisher score')
        score = masked_fisher_score(dataloader, model, current_mask = mask)
        
        # Update the mask
        print(f'\tUpdating the mask')
        mask = create_gradiend_mask(score, sparsity = s, mask_type = mask_type)
        
    return mask