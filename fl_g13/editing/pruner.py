# Iurada et al. (2024). TaLoS: Task-Localized Sparse Fine-Tuning for Efficient Transfer Learning. arXiv preprint arXiv:2401.00001.
# https://arxiv.org/html/2504.02620v1#S4.E7


import torch # Import the PyTorch library for deep learning operations
import numpy as np # Import NumPy for numerical operations (though not heavily used in this snippet)
import gc # Import garbage collection interface (not used in this snippet)
import torch.nn as nn # Import PyTorch neural network module
import torch.nn.functional as F # Import PyTorch functional module (e.g., activation functions, pooling)
from tqdm import tqdm # Import tqdm for displaying progress bars

# Custom layers and models assumed to be defined elsewhere
import layers # Import custom layers module (contains definitions for Conv2d, MultiheadAttention, Linear, LayerNorm)
from datasets.common import maybe_dictionarize # Import utility function for handling dataset items
from heads import get_classification_head # Import function to get a classification head for the model
from layers import Conv2d, MultiheadAttention, Linear # Import specific layer types from the custom layers module
from task_vectors import NonLinearTaskVector # Import NonLinearTaskVector class for task arithmetic
from modeling import ImageClassifier # Import ImageClassifier model definition

# Utility to extract mask buffers from modules
def masks(module):
    # Iterate through all named buffers within a PyTorch module
    for name, buf in module.named_buffers():
        # Check if the buffer's name contains the substring "mask"
        if "mask" in name:
            # If it's a mask buffer, yield it
            yield buf

# Determines if a module is trainable (i.e., not Identity)
def trainable(module):
    # Return True if the module is NOT an instance of nn.Identity
    return not isinstance(module, (nn.Identity))

# Determines if a module is prunable (i.e., relevant for sparse training)
# The batchnorm and residual arguments are present but not used in this specific implementation
def prunable(module, batchnorm, residual): # batchnorm and residual are unused
    # Return True if the module is an instance of the specified layer types
    return isinstance(module, (layers.MultiheadAttention, layers.Linear, layers.Conv2d, layers.LayerNorm))

# Iterator over trainable parameters
def parameters(model):
    # Iterate through all modules in the model
    # Filter modules using the 'trainable' function
    for module in filter(lambda p: trainable(p), model.modules()):
        # Iterate through the parameters of the current module (do not recurse into submodules)
        for param in module.parameters(recurse=False):
            # Yield the parameter
            yield param

# Iterator over prunable parameters, yielding (mask, param)
# The bias, batchnorm, and residual arguments are present but not used in this specific implementation
def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    # Iterate through all modules in the model
    # Filter modules using the 'prunable' function
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        # Iterate through masks and parameters of the current prunable module simultaneously
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            # Yield the mask and the corresponding parameter
            yield mask, param

# Base Pruner class for sparse training
class Pruner:
    # Constructor for the Pruner class
    def __init__(self, masked_parameters):
        # Store the list of (mask, parameter) pairs
        self.masked_parameters = list(masked_parameters)
        # Initialize a dictionary to store sensitivity or importance scores for each parameter
        self.scores = {}

    # Abstract method to calculate parameter scores (must be implemented by subclasses)
    def score(self, model, loss, dataloader, device):
        # Raise a NotImplementedError if the subclass does not implement this method
        raise NotImplementedError

    # Global thresholding of scores to compute binary masks
    # This method does not require gradient computation
    @torch.no_grad()
    def _global_mask(self, sparsity):
        # Concatenate all flattened scores from the scores dictionary into a single tensor
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        # Calculate the number of parameters to keep based on the desired sparsity
        k = int((1.0 - sparsity) * global_scores.numel())
        # Proceed only if there is at least one parameter to keep
        if k >= 1:
            # Find the k-th smallest value in the global scores (this is the threshold)
            threshold, _ = torch.kthvalue(global_scores, k)
            # Iterate through each mask and parameter pair
            for mask, param in self.masked_parameters:
                # Skip if the parameter's ID is not in the scores dictionary
                if id(param) not in self.scores: continue
                # Get the score for the current parameter and move it to the mask's device
                score = self.scores[id(param)].to(mask.device)
                # Update the mask: set to 0.1 if score <= threshold, otherwise set to 1.
                # This creates a mask where 1s indicate parameters to keep, 0.1s indicate parameters to prune (during calibration)
                mask.copy_(torch.where(score <= threshold, torch.tensor([0.1], device=mask.device), torch.tensor([1.], device=mask.device)))
                # Store a detached copy of the mask on the parameter itself as 'score' attribute
                setattr(param, 'score', mask.clone().detach().cuda())

    # Local (per-parameter) thresholding
    # This method does not require gradient computation
    @torch.no_grad()
    def _local_mask(self, sparsity):
        # Iterate through each mask and parameter pair
        for mask, param in self.masked_parameters:
            # Get the score for the current parameter
            score = self.scores[id(param)]
            # Calculate the number of elements to keep for this specific parameter
            k = int((1.0 - sparsity) * score.numel())
            # Proceed only if there is at least one element to keep for this parameter
            if k >= 1:
                # Find the k-th smallest value in the flattened scores for this parameter (local threshold)
                threshold, _ = torch.kthvalue(score.flatten(), k)
                # Update the mask: set to 0 if score <= threshold, otherwise set to 1.
                # This creates a binary mask (0 or 1)
                mask.copy_(torch.where(score <= threshold, torch.tensor([0.], device=mask.device), torch.tensor([1.], device=mask.device)))

    # Method to apply masking based on scope (global or local)
    def mask(self, sparsity, scope):
        # If scope is 'global', call the global masking method
        if scope == 'global':
            self._global_mask(sparsity)
        # If scope is 'local', call the local masking method
        elif scope == 'local':
            self._local_mask(sparsity)

    # Multiplies parameters by their binary masks to apply sparsity
    # This method does not require gradient computation
    @torch.no_grad()
    def apply_mask(self):
        # Iterate through each mask and parameter pair
        for mask, param in self.masked_parameters:
            # Perform element-wise multiplication of the parameter by its mask (in-place)
            param.mul_(mask)

    # Utility: returns count of remaining (non-zero) parameters
    def stats(self):
        # Initialize counts for remaining and total parameters
        remaining_params, total_params = 0, 0
        # Iterate through each mask and parameter pair
        for mask, param in self.masked_parameters:
            # Check if the parameter's ID is in the scores dictionary (meaning it was considered for pruning)
            if id(param) in self.scores:
                # Sum the values in the 'score' attribute (which holds the mask) to count remaining params
                remaining_params += param.score.clone().detach().cpu().numpy().sum()
            # Add the total number of elements in the mask (equal to parameter size) to total_params
            total_params += mask.numel()
        # Return the counts of remaining and total parameters
        return remaining_params, total_params

# TaLoS: Task-Localized Sparse Fine-Tuning
# Inherits from the base Pruner class
class TaLoS(Pruner):
    # Constructor for the TaLoS class
    def __init__(self, masked_parameters):
        # Call the constructor of the base Pruner class
        super(TaLoS, self).__init__(masked_parameters)
        # Initialize R, the number of forward-backward passes per batch for scoring
        self.R = 1

    # Method to calculate scores for TaLoS (based on squared gradients)
    def score(self, model, loss, dataloader, device, batch_limit):
        # Move the model to the specified device
        model = model.to(device)
        # Set the model to evaluation mode (disables dropout, batchnorm tracking, etc.)
        model.eval()

        # Enable gradients only for the masked parameters that will be scored
        for m, p in masked_parameters(model):
            # Disable gradients for mask buffers
            m.requires_grad = False
            # Enable gradients for parameters
            p.requires_grad = True
            # Initialize scores for each parameter with zeros on the CPU
            self.scores[id(p)] = torch.zeros_like(p).cpu()

        # Accumulate squared gradients (approximate Fisher Information)
        # Iterate through the dataloader with a progress bar
        for data in tqdm(dataloader):
            # Dictionarize the data if needed
            data = maybe_dictionarize(data)
            # Move the input images to the specified device
            input = data['images'].to(device)
            # Perform R forward-backward passes per batch
            for _ in range(self.R):
                # Perform a forward pass to get logits
                logits = model(input)
                # Sample an output index from the categorical distribution of logits
                outdx = torch.distributions.Categorical(logits=logits).sample().unsqueeze(1).detach()
                # Gather the sampled logits
                samples = logits.gather(1, outdx)

                # Iterate through each sample in the batch
                for idx in range(data['images'].size(0)):
                    # Zero out gradients before backward pass for each sample
                    model.zero_grad()
                    # Perform a backward pass to compute gradients for the current sample
                    # retain_graph=True is needed because we are doing backward passes sample by sample
                    torch.autograd.backward(samples[idx], retain_graph=True)
                    # Iterate through masked parameters to accumulate squared gradients
                    for m, p in masked_parameters(model):
                        # Check if the parameter requires gradients and has a gradient computed
                        if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
                            # Add the squared gradients (detached from the graph) to the scores
                            self.scores[id(p)] += p.grad.data.pow(2).detach().cpu()

        # Clean up gradients and reset requires_grad flags
        for m, p in masked_parameters(model):
            # If the parameter required gradients and has a gradient
            if p.requires_grad and hasattr(p, 'grad'):
                # Zero out the gradient data
                p.grad.data.zero_()
            # Disable gradients for mask buffers
            m.requires_grad = False
            # Enable gradients for parameters (presumably for subsequent fine-tuning)
            p.requires_grad = True

# LoTA: Lottery Ticket-Aware Pruning (post-hoc)
# Inherits from the base Pruner class
class LoTA(Pruner):
    # Constructor for the LoTA class
    def __init__(self, masked_parameters):
        # Call the constructor of the base Pruner class
        super(LoTA, self).__init__(masked_parameters)
        # Initialize epochs (not used directly in this score method)
        self.epochs = 1

    # Method to calculate scores for LoTA (based on absolute weight difference)
    def score(self, model, dataset_name, args):
        # Define paths to zero-shot and fine-tuned checkpoints
        zs_ckpt = f"{args.save}/{dataset_name}/zeroshot.pt"
        ft_ckpt = f"{args.save}/{dataset_name}/finetuned.pt"
        # Create a NonLinearTaskVector representing the difference between fine-tuned and zero-shot weights
        # Apply this task vector to the zero-shot checkpoint
        image_encoder = NonLinearTaskVector(zs_ckpt, ft_ckpt).apply_to(zs_ckpt, scaling_coef=1.0)
        # Get the classification head for the dataset
        classification_head = get_classification_head(args, dataset_name)
        # Create a fine-tuned model using the image encoder and classification head, move to device
        ft_model = ImageClassifier(image_encoder, classification_head).to(args.device)

        # Mask the pretrained ViT layers in the fine-tuned model
        layers.mask_pretrained_vit(ft_model, args.device, torch.float32, skip_ln=False)

        # Score is simply the absolute weight difference
        # Disable gradient computation for this scoring process
        with torch.no_grad():
            # Iterate through masked parameters of both the fine-tuned model and the original model
            for (mf, pf), (mp, pp) in zip(masked_parameters(ft_model), self.masked_parameters):
                # Calculate the absolute difference between the fine-tuned parameter (pf) and the original parameter (pp)
                # Store this absolute difference as the score for the original parameter (pp)
                self.scores[id(pp)] = (pf - pp).detach().abs().cuda()

