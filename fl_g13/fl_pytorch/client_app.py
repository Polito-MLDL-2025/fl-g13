import gc

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context

from fl_g13.modeling.train import train
from fl_g13.modeling.eval import eval
from fl_g13.editing import create_gradiend_mask, fisher_scores, mask_dict_to_list, compress_mask_sparse

from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.fl_pytorch.datasets import get_transforms, load_flwr_datasets

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

        self.mask = None
        if model_editing:
            self._compute_mask(sparsity=sparsity, mask_type=mask_type)

        self.model.to(self.device)

    # --- MASKING --- #

    def _compute_mask(self, sparsity, mask_type):
        scores = fisher_scores(dataloader=self.valloader, model=self.model, verbose=1, loss_fn=self.criterion)
        mask = create_gradiend_mask(class_score=scores, sparsity=sparsity, mask_type=mask_type)
        mask_list = mask_dict_to_list(self.model, mask)
        if not hasattr(self.optimizer, "set_mask"):
            raise Exception("The optimizer should have a set_mask method")
        self.optimizer.set_mask(mask_list)
        self.mask = compress_mask_sparse(mask_list)

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
        all_training_losses, _, _, _ = train(
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
        )

        updated_weights = get_weights(self.model)
        flatten_updated_weights = np.concatenate([w.flatten() for w in updated_weights])

        # Save mdoel to context's state to use in a future fit() call
        self._save_weights_to_state()

        # Client drift (Euclidean)
        drift = np.linalg.norm(flatten_updated_weights - flatten_global_weights)

         # --- Modified: Conditionally include the mask in results ---
        results = {
            "train_loss": sum(all_training_losses),
            "drift": drift.tolist(),
        }
        if self.mask is not None: # Only include 'mask' if it was computed
            results["mask"] = self.mask
        gc.collect()
        torch.cuda.empty_cache()
        return (
            updated_weights,
            len(self.trainloader.dataset),
            results, 
            # if you have more complex metrics you have to serialize them with json since Metrics value allow only Scalar
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        test_loss, test_accuracy, _ = eval(self.valloader, self.model, self.criterion)
        gc.collect()
        torch.cuda.empty_cache()

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
        sparsity=0.2
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
            sparsity=sparsity
        ).to_client()

    app = ClientApp(client_fn=client_fn)
    return app
