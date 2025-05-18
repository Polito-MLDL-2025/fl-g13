import json

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import ArrayRecord

from fl_g13.editing import create_gradiend_mask, fisher_scores, mask_dict_to_list, compress_mask_sparse
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.eval import eval
from fl_g13.modeling.train import train

class FlowerClient(NumPyClient):
    def __init__(
            self,
            client_state,
            local_epochs,
            trainloader,
            valloader,
            model,
            criterion,
            optimizer,
            scheduler=None,
            device=None,
            model_editing=False,
            sparsity=0.2,
            mask_type='global',
            is_save_weights_to_state=False,
            verbose=0
    ):
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.local_layer_name = "classification-head"
        self.last_global_weights = None
        self.mask = False
        self.mask_list = None
        self.model_editing = model_editing
        self.scheduler = scheduler
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_save_weights_to_state = is_save_weights_to_state
        self.verbose = verbose
        self.mask = None
        self.sparsity = sparsity
        self.mask_type = mask_type
        if model_editing:
            self._compute_mask(sparsity=sparsity, mask_type=mask_type)

        self.model.to(self.device)

    # --- MASKING --- #

    def _compute_mask(self, sparsity, mask_type):
        scores = fisher_scores(dataloader=self.valloader, model=self.model, verbose=1, loss_fn=self.criterion)
        mask = create_gradiend_mask(class_score=scores, sparsity=sparsity, mask_type=mask_type)
        mask_list = mask_dict_to_list(self.model, mask)
        self.mask_list = mask_list
        self.set_mask(mask_list)
        

    def set_mask(self, mask):
        if not hasattr(self.optimizer, "set_mask"):
            raise Exception("The optimizer should have a set_mask method")
        self.optimizer.set_mask(mask)
        self.mask = compress_mask_sparse(mask)

    # --- SAVE AND LOAD WEIGHTS TO STATE --- #

    def _save_weights_to_state(self):
        # Convert model state dictionary to ArrayDict
        arr_record = ArrayRecord(self.model.state_dict())

        # Add the state to the context (replace if already exists)
        self.client_state["full_model_state"] = arr_record

    def _load_weights_from_state(self):
        # Extract state from context
        state_dict = self.client_state["full_model_state"].to_torch_state_dict()

        # Apply the state found in context to the model
        self.model.load_state_dict(state_dict, strict=True)

    # --- FIT AND EVALUATE --- #

    def fit(self, parameters, config):

        # Save weights from global models
        flatten_global_weights = np.concatenate([p.flatten() for p in parameters])

        # Apply weights from global models (the whole local model weights are replaced)
        set_weights(self.model, parameters)

        # Train using the new weights
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
            verbose=self.verbose
        )

        # fine tuned weights
        updated_weights = get_weights(self.model)
        flatten_updated_weights = np.concatenate([w.flatten() for w in updated_weights])

        # Save mdoel to context's state to use in a future fit() call
        if self.is_save_weights_to_state:
            self._save_weights_to_state()

        # Client drift (Euclidean)
        drift = np.linalg.norm(flatten_updated_weights - flatten_global_weights)
        
        fit_params = updated_weights

        results = {
            "train_loss":  all_training_losses[-1],
            "drift": drift.tolist(),
        }

        if all_training_accuracies and all_training_losses:
            results["training_accuracies"] = json.dumps(all_training_accuracies)
            results["training_losses"] = json.dumps(all_training_losses)
        # --- Modified: Conditionally include the mask in results ---

        if self.mask is not None:  # Only include 'mask' if it was computed
            results["mask"] = self.mask

        return (
            fit_params,
            len(self.trainloader.dataset),
            results,
            # if you have more complex metrics you have to serialize them with json since Metrics value allow only Scalar
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        test_loss, test_accuracy, _ = eval(self.valloader, self.model, self.criterion)

        return test_loss, len(self.valloader.dataset), {"accuracy": test_accuracy}