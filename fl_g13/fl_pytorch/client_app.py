import json

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, ConfigRecord

from fl_g13.dataset import update_dataloader
from fl_g13.editing import create_gradiend_mask, fisher_scores, mask_dict_to_list, compress_mask_sparse
from fl_g13.editing.masking import uncompress_mask_sparse
from fl_g13.fl_pytorch.datasets import get_transforms, load_flwr_datasets
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.eval import eval
from fl_g13.modeling.train import train


# *** ---------------- CLIENT CLASS ---------------- *** #

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
            verbose=0,
            mask=None,
            mask_func=None,
            model_editing_batch_size=16,
    ):
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_save_weights_to_state = is_save_weights_to_state
        self.verbose = verbose
        self.mask = mask
        self.model_editing_batch_size =model_editing_batch_size
        self.mask_type = mask_type
        self.sparsity = sparsity
        # check model editing condition
        if model_editing:
            # require set_mask method to update mask to optimizer
            if not hasattr(self.optimizer, "set_mask"):
                raise Exception("Model Editting require optimizer have to implement set_mask method to update mask to itself")
            # if mask is None, the client compute the mask itself by fisher score and save to state
            if not mask:
                if mask_func and callable(mask_func):
                    ## call mask func to get mask for model editing
                    mask_func(self)
                else:
                    self._load_mask_from_state()
                    if not self.mask:
                        self._compute_mask()
                        self._save_mask_to_state()

            else:
                self.set_mask(mask)

        self.model.to(self.device)
    # --- MASKING --- #

    def _compute_mask(self):
        mask_dataloader = update_dataloader(self.trainloader, self.model_editing_batch_size)
        scores = fisher_scores(dataloader=mask_dataloader, model=self.model, verbose=1, loss_fn=self.criterion)
        mask = create_gradiend_mask(class_score=scores, sparsity=self.sparsity, mask_type=self.mask_type)
        mask_list = mask_dict_to_list(self.model, mask)
        self.set_mask(mask_list)

    def set_mask(self, mask):
        mask = [tensor.to(self.device) for tensor in mask]
        self.optimizer.set_mask(mask)
        self.mask = compress_mask_sparse(mask)

    # --- SAVE AND LOAD MASK TO STATE --- #
    def _save_mask_to_state(self):
        self.client_state["mask"] = ConfigRecord({'compress_mask':self.mask})
    def _load_mask_from_state(self):
        if self.client_state.get("mask") is None:
            return
        self.set_mask(uncompress_mask_sparse(self.client_state["mask"]['compress_mask']))

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

        updated_weights = get_weights(self.model)
        flatten_updated_weights = np.concatenate([w.flatten() for w in updated_weights])

        # Save mdoel to context's state to use in a future fit() call
        if self.is_save_weights_to_state:
            self._save_weights_to_state()

        # Client drift (Euclidean)
        drift = np.linalg.norm(flatten_updated_weights - flatten_global_weights)

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
            updated_weights,
            len(self.trainloader.dataset),
            results,
            # if you have more complex metrics you have to serialize them with json since Metrics value allow only Scalar
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        test_loss, test_accuracy, _ = eval(self.valloader, self.model, self.criterion)

        return test_loss, len(self.valloader.dataset), {"accuracy": test_accuracy}


# *** ---------------- CLIENT APP ---------------- *** #

def load_client_dataloaders(
        context: Context,
        partition_type,
        num_shards_per_partition,
        batch_size,
        train_test_split_ratio,
        transform=get_transforms
):
    # Retrive meta-data from context
    partition_id = context.node_config["partition-id"]  # assigned at runtime
    num_partitions = context.node_config["num-partitions"]

    # Load flower datasets for clients
    trainloader, valloader = load_flwr_datasets(
        partition_id=partition_id,
        partition_type=partition_type,
        num_partitions=num_partitions,
        num_shards_per_partition=num_shards_per_partition,
        batch_size=batch_size,
        train_test_split_ratio=train_test_split_ratio,
        transform=transform
    )
    return trainloader, valloader


def get_client_app(
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        load_data_fn=load_client_dataloaders,
        batch_size=50,
        partition_type="iid",
        num_shards_per_partition=2,
        train_test_split_ratio=0.2,
        local_epochs=4,
        model_editing=False,
        mask_type='global',
        sparsity=0.2,
        is_save_weights_to_state=False,
        verbose=0,
        mask=None,
        mask_func=None,
        model_editing_batch_size=16
) -> ClientApp:
    def client_fn(context: Context):
        print(f"[Client] Client on device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"[Client] CUDA available in client: {torch.cuda.is_available()}")

        trainloader, valloader = load_data_fn(
            context=context,
            partition_type=partition_type,
            batch_size=batch_size,
            num_shards_per_partition=num_shards_per_partition,
            train_test_split_ratio=train_test_split_ratio
        )
        client_state = context.state
        return FlowerClient(
            client_state=client_state,
            local_epochs=local_epochs,
            trainloader=trainloader,
            valloader=valloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_editing=model_editing,
            mask_type=mask_type,
            sparsity=sparsity,
            is_save_weights_to_state=is_save_weights_to_state,
            verbose=verbose,
            mask=mask,
            mask_func=mask_func,
            model_editing_batch_size=model_editing_batch_size
        ).to_client()

    app = ClientApp(client_fn=client_fn)
    return app
