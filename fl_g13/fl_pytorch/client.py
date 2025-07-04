import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from flwr.client import NumPyClient
from flwr.common import ArrayRecord, ConfigRecord, NDArrays, Scalar
from torch.utils.data import DataLoader

from fl_g13.dataset import update_dataloader
from fl_g13.editing import (
    compress_mask_sparse,
    create_mask,
    mask_dict_to_list,
    uncompress_mask_sparse,
)
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling import train, eval


class CustomNumpyClient(NumPyClient):
    """
    A custom Flower NumPyClient for federated learning with model editing capabilities.

    This client handles local training, evaluation, and optional model editing
    through parameter masking. It can compute, apply, and persist masks and
    model weights across federated learning rounds.
    """

    def __init__(
        self,
        client_state: Dict[str, Any],
        local_epochs: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        local_steps: Optional[int] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        model_editing: bool = False,
        sparsity: float = 0.8,
        mask_type: str = "global",
        is_save_weights_to_state: bool = False,
        verbose: int = 0,
        mask: Optional[List[torch.Tensor]] = None,
        mask_calibration_round: int = 1,
        model_editing_batch_size: int = 1,
        mask_func: Optional[Callable[["CustomNumpyClient"], None]] = None,
    ) -> None:
        """
        Initializes the CustomNumpyClient.

        Args:
            client_state (Dict[str, Any]): A dictionary to persist state across rounds.
            local_epochs (int): The number of local training epochs.
            trainloader (DataLoader): The DataLoader for the training set.
            valloader (DataLoader): The DataLoader for the validation set.
            model (torch.nn.Module): The PyTorch model to be trained.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            local_steps (Optional[int]): The number of local training steps. If set, overrides local_epochs.
            scheduler (Optional[Any]): The learning rate scheduler.
            device (Optional[torch.device]): The device to run the model on.
            model_editing (bool): If True, enables model editing with masks.
            sparsity (float): The target sparsity for the mask.
            mask_type (str): The type of mask to compute ('global' or 'local').
            is_save_weights_to_state (bool): If True, saves model weights to the client state.
            verbose (int): The verbosity level.
            mask (Optional[List[torch.Tensor]]): An initial mask to apply.
            mask_calibration_round (int): The number of rounds for mask calibration.
            model_editing_batch_size (int): The batch size for mask computation.
            mask_func (Optional[Callable]): A function to compute the mask.
        """
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.local_steps = local_steps
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.is_save_weights_to_state = is_save_weights_to_state
        self.verbose = verbose
        self.mask = mask
        self.mask_calibration_round = mask_calibration_round
        self.model_editing_batch_size = model_editing_batch_size
        self.mask_type = mask_type
        self.sparsity = sparsity

        # --- Model Editing Setup ---
        if model_editing:
            # The optimizer must have a `set_mask` method for model editing
            if not hasattr(self.optimizer, "set_mask"):
                raise AttributeError(
                    "Model editing requires the optimizer to have a `set_mask` method."
                )

            # If no mask is provided, compute or load it
            if not mask:
                if mask_func and callable(mask_func):
                    # Use the provided function to compute the mask
                    mask_func(self)
                else:
                    # Otherwise, try to load from state or compute a new one
                    self._load_mask_from_state()
                    if not self.mask:
                        self._compute_mask(sparsity, mask_type)
                        self._save_mask_to_state()
            else:
                self.set_mask(mask)

        self.model.to(self.device)

    def _compute_mask(self, sparsity: float, mask_type: str) -> None:
        """Computes a new mask based on Fisher information and applies it."""
        if self.verbose > 0:
            print(f"Client computing mask with sparsity {sparsity} and type {mask_type}")
        mask_dataloader = update_dataloader(
            self.trainloader, self.model_editing_batch_size
        )
        mask_dict = create_mask(
            model=self.model,
            dataloader=mask_dataloader,
            sparsity=sparsity,
            mask_type=mask_type,
            rounds=self.mask_calibration_round,
            verbose=self.verbose,
        )
        mask_list = mask_dict_to_list(self.model, mask_dict)
        self.set_mask(mask_list)

    def set_mask(self, mask: List[torch.Tensor]) -> None:
        """Applies a mask to the optimizer and stores the compressed version."""
        mask_on_device = [tensor.to(self.device) for tensor in mask]
        self.optimizer.set_mask(mask_on_device)
        self.mask = compress_mask_sparse(mask_on_device)

    def _save_mask_to_state(self) -> None:
        """Saves the compressed mask to the client's state."""
        if self.mask is not None:
            self.client_state["mask"] = ConfigRecord({"compress_mask": self.mask})

    def _load_mask_from_state(self) -> None:
        """Loads and applies a mask from the client's state if it exists."""
        if "mask" in self.client_state:
            compressed_mask = self.client_state["mask"]["compress_mask"]
            self.set_mask(uncompress_mask_sparse(compressed_mask))
            if self.verbose > 0:
                print("Client loaded mask from state.")

    def _save_weights_to_state(self) -> None:
        """Saves the model's state dictionary to the client's state."""
        arr_record = ArrayRecord(self.model.state_dict())
        self.client_state["full_model_state"] = arr_record

    def _load_weights_from_state(self) -> None:
        """Loads model weights from the client's state if they exist."""
        if "full_model_state" in self.client_state:
            state_dict = self.client_state["full_model_state"].to_torch_state_dict()
            self.model.load_state_dict(state_dict, strict=True)
            if self.verbose > 0:
                print("Client loaded weights from state.")

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Performs local training on the client.

        Args:
            parameters (NDArrays): The global model parameters from the server.
            config (Dict[str, Scalar]): Configuration parameters from the server.

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]:
                - The updated local model parameters.
                - The number of examples used for training.
                - A dictionary of training metrics.
        """
        set_weights(self.model, parameters)

        # Perform local training
        all_training_losses, _, all_training_accuracies, _ = train(
            checkpoint_dir=None,
            name=None,
            start_epoch=1,
            num_epochs=self.local_epochs,
            save_every=None,
            backup_every=None,
            train_dataloader=self.trainloader,
            val_dataloader=None,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            eval_every=None,
            verbose=self.verbose,
            num_steps=self.local_steps,
        )

        updated_weights = get_weights(self.model)

        if self.is_save_weights_to_state:
            self._save_weights_to_state()

        results = {}
        if all_training_losses:
            results["train_loss"] = all_training_losses[-1]
            results["training_losses"] = json.dumps(all_training_losses)
        if all_training_accuracies:
            results["training_accuracies"] = json.dumps(all_training_accuracies)

        # Include the mask in the results if it exists
        if self.mask is not None:
            results["mask"] = self.mask

        return updated_weights, len(self.trainloader.dataset), results

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Performs local evaluation on the client.

        Args:
            parameters (NDArrays): The global model parameters from the server.
            config (Dict[str, Scalar]): Configuration parameters from the server.

        Returns:
            Tuple[float, int, Dict[str, Scalar]]:
                - The evaluation loss.
                - The number of examples used for evaluation.
                - A dictionary of evaluation metrics.
        """
        set_weights(self.model, parameters)
        test_loss, test_accuracy, _ = eval(
            self.valloader, self.model, self.criterion, self.verbose
        )

        return float(test_loss), len(self.valloader.dataset), {"accuracy": float(test_accuracy)}
