import torch

def create_gradiend_mask(class_score, sparsity = 0.2):
    """
    class_score: dict of {param_name: tensor}
    sparsity: fraction of parameters to keep editable (lowest scores values)

    Returns: dict {param_name: binary_mask_tensor}
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

def mask_dict_to_list(model, gradient_mask_dict):
    """
    Converts a {param_name: mask_tensor} dict into a list of mask_tensors
    in the same order as model.parameters().
    """
    mask_list = []
    for name, param in model.named_parameters():
        if name not in gradient_mask_dict:
            raise KeyError(f"Missing mask for parameter: {name}")
        mask = gradient_mask_dict[name]
        if param.shape != mask.shape:
            raise ValueError(f"Mask shape mismatch for {name}: {param.shape} vs {mask.shape}")
        mask_list.append(mask.to(param.device))
    return mask_list
