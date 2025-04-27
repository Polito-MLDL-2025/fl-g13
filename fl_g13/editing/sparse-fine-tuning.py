# Iurada et al. (2024). TaLoS: Task-Localized Sparse Fine-Tuning for Efficient Transfer Learning. arXiv preprint arXiv:2401.00001.
# https://arxiv.org/html/2504.02620v1#S4.E7


import os # Import the os module for interacting with the operating system (e.g., file paths, directories)
import time # Import the time module for time-related functions (e.g., measuring execution time)

import torch # Import the PyTorch library for deep learning operations

from args import parse_arguments # Import a function to parse command-line arguments from a custom 'args' module
from datasets.common import get_dataloader, maybe_dictionarize # Import utilities for data loading and handling from 'datasets.common'
from datasets.registry import get_dataset # Import a function to get dataset objects from 'datasets.registry'
from distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp # Import utilities for Distributed Data Parallel (DDP) training
from eval import eval_single_dataset # Import a function for evaluating the model on a single dataset
from heads import get_classification_head # Import a function to get a classification head
from linearize import LinearizedImageEncoder # Import a class for a linearized image encoder model
from modeling import ImageClassifier, ImageEncoder # Import model classes for ImageClassifier and ImageEncoder
from utils import LabelSmoothing, cosine_lr # Import utility functions like LabelSmoothing loss and cosine learning rate scheduler
import layers # Import a custom module named 'layers' (likely contains custom layer definitions)
import pruners # Import a custom module named 'pruners' (likely contains the Pruner class and its subclasses like TaLoS)
import optimizers # Import a custom module named 'optimizers' (likely contains custom optimizer implementations like AdaptW)
import tqdm # Import tqdm for displaying progress bars
import torch.nn.functional as F # Import PyTorch functional module (though not explicitly used in this snippet's main logic)
from task_vectors import NonLinearTaskVector # Import a class related to task vectors (for loading pre/post fine-tuned models)
import copy # Import the copy module (not explicitly used in this snippet's main logic)

# Main fine-tuning function
def finetune(rank, args):
    # Set up Distributed Data Parallel (DDP) for distributed training
    setup_ddp(rank, args.world_size, port=args.port)

    # Get the name of the training dataset from arguments
    train_dataset = args.train_dataset
    # Define the checkpoint directory path
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if linearized fine-tuning is enabled based on arguments
    linearized_finetuning = args.finetuning_mode == "linear"
    # Print a message if linearized fine-tuning is used
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # Check if checkpoints already exist to potentially skip fine-tuning
    # Define the expected path for the fine-tuned model checkpoint
    ft_path = (
        os.path.join(args.save, train_dataset, "finetuned.pt")
        if args.finetuning_mode == "standard"
        else os.path.join(args.save, train_dataset, f"{args.finetuning_mode}_finetuned.pt")
    )
    # Define the expected path for the zero-shot model checkpoint
    zs_path = (
        os.path.join(args.save, train_dataset, "zeroshot.pt")
        if args.finetuning_mode == "standard"
        else os.path.join(args.save, train_dataset, f"{args.finetuning_mode}_zeroshot.pt")
    )
    # If both checkpoints exist, print a message and return their paths, skipping fine-tuning
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    # Assert that a training dataset is provided
    assert train_dataset is not None, "Please provide a training dataset."

    # Load a pre-existing task vector if specified
    if args.load is not None:
        print(f'[LOADING] {args.load}')
        # Define paths to the zero-shot and fine-tuned components of the task vector
        pt_ckpt = f'{args.save}/{train_dataset}/{args.load}zeroshot.pt'
        ft_ckpt = f'{args.save}/{train_dataset}/{args.load}finetuned.pt'
        # Create an image encoder by applying the non-linear task vector to the pre-trained checkpoint
        image_encoder = NonLinearTaskVector(pt_ckpt, ft_ckpt).apply_to(pt_ckpt, 1.0)
    else:
        # Otherwise, build a new image encoder (either linearized or standard)
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    # Get the classification head for the training dataset
    classification_head = get_classification_head(args, train_dataset)

    # Combine the image encoder and classification head to create the full model
    model = ImageClassifier(image_encoder, classification_head)

    # Freeze the classification head's parameters
    model.freeze_head()
    # Move the model to the GPU (CUDA)
    model = model.cuda()

    # Section related to Pruning
    #! Prune ==============
    # Get the training preprocessing function from the model
    preprocess_fn = model.train_preprocess
    # Define how often to print during pruning (not directly used in the pruning loop itself)
    print_every = 100
    # Set a base batch size for pruning data loading
    prune_bs = 64
    # Adjust pruning batch size based on the model size
    if args.model == 'ViT-B-16':
        prune_bs = 16
    if args.model == 'ViT-L-14':
        prune_bs = 2

    # Load the dataset for pruning (using the validation split as the pruning set, as mentioned in the paper)
    dataset = get_dataset(
        train_dataset, # Use the training dataset name
        preprocess_fn, # Use the training preprocessing function
        location=args.data_location, # Data storage location
        batch_size=prune_bs, # Batch size for pruning data
        num_workers=4, # Number of worker processes for data loading
    )
    # Get the dataloader for the pruning dataset (set to is_train=False as it's used for scoring, not training)
    data_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    # Get the number of batches in the training loader (this seems inconsistent, should likely be data_loader)
    num_batches = len(dataset.train_loader)

    # Define the number of rounds for iterative pruning mask refinement
    ROUNDS = 4
    # Define the number of batches to use for pruning score calculation (-1 likely means use all batches)
    N_PRUNING_BATCHES = -1

    # Apply initial masking to pretrained ViT layers (likely setting up mask buffers)
    layers.mask_pretrained_vit(model, args.device, torch.float32, skip_ln=False)

    # Move the model to the specified device (redundant with model.cuda() above if device is cuda)
    model = model.to(args.device)

    # Initialize the Pruner (specifically TaLoS) with the model's masked parameters
    pruner = pruners.TaLoS(pruners.masked_parameters(model))

    # Calculate the target sparsity level (1.0 - desired density)
    sparsity = 1.0 - args.sparsity
    # Define a threshold for zeroing out parameters based on their score (used later)
    zeros_thresh = 1.0
    # Set the number of forward-backward passes per batch for TaLoS scoring
    pruner.R = args.R

    # Enable masking mode for relevant layers in the model
    for layer in model.modules():
        if hasattr(layer, 'masking'):
            layer.masking = True

    # Perform the pruning mask calculation if sparsity is less than 1.0 (i.e., not keeping all parameters)
    if sparsity < 1.0:
        # Iterate through the specified number of pruning rounds
        for round in range(ROUNDS):
            # Calculate the target sparsity for the current round (exponentially increasing)
            sparse = sparsity**((round + 1) / ROUNDS)
            # Print the target sparsity for the current round
            print('[+] Target sparsity:', sparse)
            # Calculate the parameter scores using the pruner's score method (TaLoS.score)
            pruner.score(model, None, data_loader, args.device, N_PRUNING_BATCHES)
            # Define the masking mode to use ('global_copy' in this case)
            mode = 'global_copy'
            # Apply the mask based on the calculated scores and the current round's sparsity
            pruner.mask(sparse, mode)

    # Get statistics about the remaining (non-zero) parameters after masking
    remaining_params, total_params = pruner.stats()
    # Print the pruning statistics
    print(f'{int(remaining_params)} / {int(total_params)} | {remaining_params / total_params}')

    # Section related to preparing the model for training after pruning
    # Disable gradient computation for this block
    with torch.no_grad():
        # Initialize counters for total and zeroed parameters
        tot, cnt = 0, 0
        # Iterate through all modules in the model
        for layer in model.modules():

            # If the layer has a 'masking' attribute (meaning it was considered for pruning)
            if hasattr(layer, 'masking'):
                # Disable masking mode for the layer
                layer.masking = False

                # Iterate through the layer's buffers
                for k in layer._buffers:
                    # If the buffer name contains 'mask' and the buffer is not None
                    if 'mask' in k and layer._buffers[k] is not None:
                        # Move the mask buffer to CPU
                        layer._buffers[k] = layer._buffers[k].cpu()
                        # Set the mask buffer to None (releasing memory)
                        layer._buffers[k] = None

                # Iterate through the layer's named parameters
                for name, param in layer.named_parameters():
                    # If the parameter has a 'score' attribute (which stores the final mask from pruner.mask)
                    if hasattr(param, 'score'):
                        # Calculate the percentage of elements in the score (mask) that are 0.0
                        zeros_pctg = param.score[param.score == 0.0].numel() / param.score.numel()
                        # If the percentage of zeros meets or exceeds the zeros_thresh (1.0)
                        if zeros_pctg >= zeros_thresh:
                            # Disable gradient computation for this parameter (freeze it)
                            param.requires_grad_(False)
                            # Move the parameter's score (mask) to CPU
                            param.score = param.score.to('cpu')
                            # Delete the 'score' attribute from the parameter
                            delattr(param, 'score')
                            # Increment the count of frozen parameters
                            cnt += 1
                        else:
                            # If the parameter was not completely zeroed out by the mask
                            # Move the parameter's score (mask) to the device (likely GPU)
                            param.score = param.score.to(args.device)
                        # Increment the total count of parameters processed in this block
                        tot += 1

            # If the layer does not have children modules (it's a leaf module)
            elif len(list(layer.children())) == 0:
                # Iterate through the layer's named parameters
                for name, param in layer.named_parameters():
                    # Disable gradient computation for this parameter (freeze it)
                    param.requires_grad_(False)

        # Print the percentage of parameters that were frozen
        print(f'Frozen {cnt} / {tot} params. ({100 * cnt / tot:.2f}%)')

    # Empty the CUDA cache for memory management
    torch.cuda.empty_cache()

    # Section related to Model Training after Pruning
    #! Train ==============
    # Get the training preprocessing function again (should be the same as before)
    preprocess_fn = model.train_preprocess
    # Define how often to print during training
    print_every = 100

    # Load the dataset for training
    dataset = get_dataset(
        train_dataset, # Use the training dataset name
        preprocess_fn, # Use the training preprocessing function
        location=args.data_location, # Data storage location
        batch_size=args.batch_size, # Batch size for training
        num_workers=5, # Number of worker processes for data loading
    )
    # Get the dataloader for the training dataset (set to is_train=True)
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    # Get the number of batches in the training loader
    num_batches = len(dataset.train_loader)

    # Distribute the dataloader for DDP training
    ddp_loader = distribute_loader(data_loader)
    # Assign the model to ddp_model (will be wrapped by DDP later if world_size > 1)
    ddp_model = model

    # Define the loss function (CrossEntropyLoss, optionally with LabelSmoothing)
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Get a list of parameters that require gradients (these are the trainable ones after pruning)
    params = [p for p in ddp_model.parameters() if p.requires_grad]

    # Print a separator and optimizer information
    print("=" * 100)
    print("Using [AdaptW] Optimizer")
    print("=" * 100)
    # Initialize the optimizer (AdaptW) with the trainable parameters and hyperparameters
    optimizer = optimizers.AdaptW(params, lr=args.lr, weight_decay=args.wd)

    # Initialize the learning rate scheduler (cosine annealing)
    scheduler = cosine_lr(
        optimizer, # The optimizer to schedule
        args.lr, # Base learning rate
        args.warmup_length, # Number of warmup steps
        args.epochs * num_batches // args.num_grad_accumulation, # Total number of training steps
    )
    # Initialize the gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Save the zero-shot model checkpoint if saving is enabled and it's the main process
    if args.save is not None and is_main_process():
        # Create the checkpoint directory if it doesn't exist
        os.makedirs(ckpdir, exist_ok=True)
        # Define the path for the zero-shot checkpoint
        model_path = (
            os.path.join(ckpdir, "zeroshot.pt")
            if args.finetuning_mode == "standard"
            else os.path.join(ckpdir, f"{args.finetuning_mode}_zeroshot.pt")
        )
        # Save the image encoder part of the model
        ddp_model.image_encoder.save(model_path)

    # Main training loop
    for epoch in range(args.epochs):
        # Set the model to training mode
        ddp_model.train()

        # Iterate through batches in the distributed dataloader
        for i, batch in enumerate(ddp_loader):
            # Record the start time of the batch processing
            start_time = time.time()

            # Calculate the current training step number
            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            # Dictionarize the batch if needed
            batch = maybe_dictionarize(batch)
            # Move inputs and labels to the GPU
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            # Record the time taken for data loading
            data_time = time.time() - start_time

            # Use mixed precision (autocast) for the forward pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                # Perform a forward pass
                logits = ddp_model(inputs)
                # Calculate the loss, scaled by the gradient accumulation factor
                loss = loss_fn(logits, labels) / args.num_grad_accumulation

            # Scale the loss and perform the backward pass to compute gradients
            scaler.scale(loss).backward()

            # Perform optimizer step and scheduler step only after accumulating gradients for num_grad_accumulation batches
            if (i + 1) % args.num_grad_accumulation == 0:
                # Update the learning rate using the scheduler
                scheduler(step)

                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(params, 1.0)

                # Perform the optimizer step (updates model parameters)
                scaler.step(optimizer)
                # Zero out gradients after the optimizer step
                optimizer.zero_grad(set_to_none=True)
                # Update the gradient scaler
                scaler.update()

            # Record the total time taken for the batch (including forward, backward, and optimizer step)
            batch_time = time.time() - start_time

            # Save a checkpoint periodically if enabled and it's the main process
            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                # Define the path for the checkpoint
                model_path = (
                    os.path.join(ckpdir, f"checkpoint_{step}.pt")
                    if args.finetuning_mode == "standard"
                    else os.path.join(ckpdir, f"{args.finetuning_mode}_checkpoint_{step}.pt")
                )
                # Save the image encoder part of the model (accessing the module attribute in DDP)
                ddp_model.module.image_encoder.save(model_path)

            # Print training progress periodically if it's the main process
            if (
                i % print_every == 0
                and is_main_process()
            ):
                # Calculate percentage of epoch complete
                percent_complete = 100 * i / len(ddp_loader)
                # Print training statistics
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True, # Ensure output is immediately printed
                )

    # After training, evaluate the model on the training dataset if it's the main process
    if is_main_process():
        # Get the image encoder from the DDP model
        image_encoder = ddp_model.image_encoder
        # Evaluate the model
        eval_single_dataset(image_encoder, train_dataset, args)

    # Save the final fine-tuned model checkpoint if saving is enabled and it's the main process
    if args.save is not None and is_main_process():
        # Define the path for the zero-shot checkpoint (likely already saved, but path defined again)
        zs_path = (
            os.path.join(ckpdir, f"zeroshot.pt")
            if args.finetuning_mode == "standard"
            else os.path.join(ckpdir, f"{args.finetuning_mode}_zeroshot.pt")
        )
        # Define the path for the fine-tuned checkpoint
        ft_path = (
            os.path.join(ckpdir, f"finetuned.pt")
            if args.finetuning_mode == "standard"
            else os.path.join(ckpdir, f"{args.finetuning_mode}_finetuned.pt")
        )
        # Save the image encoder part of the model as the fine-tuned checkpoint
        image_encoder.save(ft_path)
        # Return the paths to the zero-shot and fine-tuned checkpoints
        return zs_path, ft_path

    # Clean up DDP processes
    cleanup_ddp()


# Main execution block
if __name__ == "__main__":
    # Define the list of training datasets
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN"
    ]
    # Define the number of epochs for each dataset
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
    }

    # Loop through each dataset for fine-tuning
    for dataset in train_datasets:
        # Parse command-line arguments for the current run
        args = parse_arguments()

        # HACK: Some command line arguments are overwritten by defaults here.
        # Set the learning rate
        args.lr = 1e-5
        # Set the number of epochs based on the dataset
        args.epochs = epochs[dataset]
        # Set the training dataset name (appending "Val" suggests using the validation split for training data loading setup, which might be a typo or specific data handling)
        args.train_dataset = dataset + "Val"
        # Set the batch size
        args.batch_size = 128
        # Set the number of gradient accumulation steps
        args.num_grad_accumulation = 1

        # Print separator and information about the current fine-tuning task
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        # Launch distributed training processes (one process per world_size)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
