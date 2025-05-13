"""pytorch-example: A Flower / PyTorch app."""

from typing import List

# from fl_g13.fl_pytorch.task import train, test
import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, RecordDict
from torch import nn, Tensor
from torch.utils.data import DataLoader

from fl_g13.editing import fisher_scores, create_gradiend_mask, mask_dict_to_list
from fl_g13.editing.masking import compress_mask_sparse
from fl_g13.fl_pytorch.datasets import load_datasets, get_transforms
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.eval import eval
from fl_g13.modeling.train import train


class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.

    """

    def __init__(
            self, model: nn.Module, client_state: RecordDict, trainloader: DataLoader, valloader: DataLoader,
            local_epochs: int,
            optimizer: torch.optim.Optimizer = None, criterion=None, scheduler=None,
            device=None,
            model_editing=False,
            mask_type='global',
            sparsity=0.2
    ):
        self.model = model
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()
        if not optimizer:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.local_layer_name = "classification-head"
        self.last_global_weights = None
        self.mask = False
        self.mask_list = None
        self.model_editing = model_editing
        if model_editing:
            self._compute_mask(sparsity=sparsity, mask_type=mask_type)

    def set_mask(self, mask: List[Tensor]):
        if not hasattr(self.optimizer, "set_mask"):
            raise Exception("The optimizer should have a set_mask method")
        self.optimizer.set_mask(mask)
        self.mask = compress_mask_sparse(mask)

    def _compute_mask(self, sparsity=0.2, mask_type='global'):
        scores = fisher_scores(dataloader=self.valloader, model=self.model, verbose=1, loss_fn=self.criterion)
        mask = create_gradiend_mask(class_score=scores, sparsity=sparsity, mask_type=mask_type)
        mask_list = mask_dict_to_list(self.model, mask)
        self.mask_list = mask_list
        self.set_mask(mask_list)

    def _compute_task_vector(self, updated_weights, pre_trained_weights):
        """compute τ = (θ* − θ₀) ⊙ mask"""
        fine_tuned_weights_tensors = [torch.tensor(w, device=self.device) for w in updated_weights]
        pre_trained_weights_tensors = [torch.tensor(w, device=self.device) for w in pre_trained_weights]
        task_vector = [
            mask_layer * (fine_tuned_layer - pre_trained_layer)
            for fine_tuned_layer, pre_trained_layer, mask_layer in zip(
                fine_tuned_weights_tensors, 
                pre_trained_weights_tensors, 
                self.mask_list
            )
        ]
        # Convert to type required by Flower
        fit_params = [layer.cpu().numpy() for layer in task_vector]
        return fit_params
        

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # pre-trained weigths
        self.last_global_weights = model_weights_to_vector(parameters)

        # Apply weights from global models (the whole local model weights are replaced)
        set_weights(self.model, parameters)

        # sparse-fine tuning on client specific task
        all_training_losses, _, _, _ = train(checkpoint_dir=None,
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

        # fine tuned weights
        updated_weights = get_weights(self.model)

        updated_vector = model_weights_to_vector(updated_weights)
                     

        # Client drift (Euclidean)
        drift = np.linalg.norm(updated_vector - self.last_global_weights)
        # Save classification head to context's state to use in a future fit() call
        self._save_layer_weights_to_state()

        if self.model_editing:
            fit_params = self._compute_task_vector(updated_weights, parameters)
        else:
            fit_params = updated_weights


        return (
            fit_params,
            len(self.trainloader.dataset),
            {
                "train_loss": sum(all_training_losses), 
                "drift": drift.tolist(),
                'mask':self.mask,
            }
            # if you have more complex metrics you have to serialize them with json since Metrics value allow only Scalar
        )

    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        arr_record = ArrayRecord(self.model.state_dict())

        # Add to RecordDict (replace if already exists)
        self.client_state[self.local_layer_name] = arr_record

    def _load_layer_weights_from_state(self):
        """Load last layer weights to state."""
        if self.local_layer_name not in self.client_state.array_records:
            return

        state_dict = self.client_state[self.local_layer_name].to_torch_state_dict()

        # apply previously saved classification head by this client
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        """
        set_weights(self.model, parameters)
        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        self._load_layer_weights_from_state()
        test_loss, test_accuracy, _ = eval(self.valloader, self.model, self.criterion)

        return test_loss, len(self.valloader.dataset), {"accuracy": test_accuracy}


def model_weights_to_vector(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])


def load_data_client_default(context: Context,
                             partition_type="iid",
                             batch_size=50,
                             num_shards_per_partition=2,
                             train_test_split_ratio=0.2,
                             transfrom=get_transforms
                             ):
    partition_id = context.node_config["partition-id"]  # assigned at runtime
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_datasets(
        partition_id,
        num_partitions,
        partition_type=partition_type,
        batch_size=batch_size,
        num_shards_per_partition=num_shards_per_partition,
        train_test_split_ratio=train_test_split_ratio,
        transform=transfrom
    )
    return trainloader, valloader


def get_client_app(
        load_data_fn=load_data_client_default,
        model=None,
        optimizer=None,
        criterion=None,
        device=None,
        partition_type="iid",
        local_epochs=4,
        batch_size=50,
        num_shards_per_partition=2,
        scheduler=None,
        train_test_split_ratio=0.2,
        model_editing=False,
        mask_type='global',
        sparsity=0.2
) -> ClientApp:
    """Create a Flower client app."""

    def client_fn(context: Context):
        """Create a Flower client."""

        trainloader, valloader = load_data_fn(
            context=context,
            partition_type=partition_type,
            batch_size=batch_size,
            num_shards_per_partition=num_shards_per_partition,
            train_test_split_ratio=train_test_split_ratio
        )
        # Return Client instance
        # We pass the state to persist information across
        # participation rounds. Note that each client always
        # receives the same Context instance (it's a 1:1 mapping)
        client_state = context.state
        return FlowerClient(
            model=model,
            client_state=client_state,
            trainloader=trainloader,
            valloader=valloader,
            local_epochs=local_epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            model_editing=model_editing,
            mask_type=mask_type,
            sparsity=sparsity
        ).to_client()

    app = ClientApp(client_fn=client_fn)
    return app
