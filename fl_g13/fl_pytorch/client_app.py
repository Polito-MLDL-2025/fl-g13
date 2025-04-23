"""pytorch-example: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, RecordDict

from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.fl_pytorch.datasets import load_datasets
#from fl_g13.fl_pytorch.task import train, test
import numpy as np
from typing import List
from fl_g13.modeling.eval import eval
from fl_g13.modeling.train import train


class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
            self, model, client_state: RecordDict, trainloader, valloader,
            local_epochs,
            optimizer=None, criterion=None, scheduler=None,
            device=None
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
        self.scheduler = scheduler
        self.local_layer_name = "classification-head"
        self.last_global_weights = None

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # config is a dict with the configuration metrics elaborated by strategy's on_fit_config_fn callback
        lr = config.get("lr")

        self.last_global_weights = model_weights_to_vector(parameters)

        # Apply weights from global models (the whole local model weights are replaced)
        set_weights(self.model, parameters)

        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        # self._load_layer_weights_from_state()

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
        
        updated_weights = get_weights(self.model)
        updated_vector = model_weights_to_vector(updated_weights)

        # Client drift (Euclidean)
        drift = np.linalg.norm(updated_vector - self.last_global_weights)
        # Save classification head to context's state to use in a future fit() call
        self._save_layer_weights_to_state()

        # Return locally-trained model and metrics
        return (
            updated_weights,
            len(self.trainloader.dataset),
            {"train_loss": sum(all_training_losses), "drift": drift.tolist()}, # if you have more complex metrics you have to serialize them with json since Metrics value allow only Scalar
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

def get_client_app( 
        model=None, 
        optimizer=None, 
        criterion=None,
        device=None, 
        partition="iid",
        local_epochs=1,
    ) -> ClientApp:
    """Create a Flower client app."""

    def client_fn(context: Context):
        """Create a Flower client."""
        partition_id = context.node_config["partition-id"] # assigned at runtime
        num_partitions = context.node_config["num-partitions"]
        trainloader, valloader = load_datasets(partition_id, num_partitions, partitionType=partition)

        # Return Client instance
        # We pass the state to persist information across
        # participation rounds. Note that each client always
        # receives the same Context instance (it's a 1:1 mapping)
        client_state = context.state
        return FlowerClient(
            model, 
            client_state, 
            trainloader, 
            valloader, 
            local_epochs, 
            optimizer=optimizer, 
            criterion=criterion,device=device
        ).to_client()
    
    app = ClientApp(client_fn=client_fn)
    return app
