"""pytorch-example: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, RecordDict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl_g13 import dataset as dataset_handler
from fl_g13.config import RAW_DATA_DIR
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
        # self.net.to(self.device)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        if not optimizer:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # Apply weights from global models (the whole model is replaced)
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
        # Save classification head to context's state to use in a future fit() call
        self._save_layer_weights_to_state()

        # Return locally-trained model and metrics
        return (
            get_weights(self.model),
            len(self.trainloader.dataset),
            {"train_loss": sum(all_training_losses)},
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


def get_default_data(partition_id, num_partitions, train_ratio=0.8):
    global clients_dataset_train
    global clients_dataset_val
    if not clients_dataset_train or not clients_dataset_val:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        cifar100_train = datasets.CIFAR100(root=RAW_DATA_DIR, train=True, download=True, transform=transform)
        train_dataset, val_dataset = dataset_handler.train_test_split(cifar100_train, train_ratio=train_ratio)
        # I.I.D Sharding Split
        ## k client
        clients_dataset_train = dataset_handler.iid_sharding(train_dataset, num_partitions)
        clients_dataset_val = dataset_handler.iid_sharding(val_dataset, num_partitions)
    return DataLoader(clients_dataset_train[partition_id]), DataLoader(clients_dataset_val[partition_id])


def load_data_client_default(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_ratio = context.node_config.get("train_ratio") or 0.8
    trainloader, valloader = get_default_data(partition_id, num_partitions, train_ratio)
    return trainloader, valloader


def get_client_app(load_data_fn=load_data_client_default, model=None, optimizer=None, criterion=None, device=None,
                   config: dict = {'local-epochs': 2}):
    def client_fn(context: Context):
        # Load model and data
        # net = model
        # partition_id = context.node_config["partition-id"]
        # num_partitions = context.node_config["num-partitions"]
        # trainloader, valloader = load_data(partition_id, num_partitions)
        trainloader, valloader = load_data_fn(context)
        local_epochs = context.run_config.get("local-epochs") or config.get('local-epochs')

        # Return Client instance
        # We pass the state to persist information across
        # participation rounds. Note that each client always
        # receives the same Context instance (it's a 1:1 mapping)
        client_state = context.state
        return FlowerClient(
            model, client_state, trainloader, valloader, local_epochs, optimizer=optimizer, criterion=criterion,
            device=device
        ).to_client()

    # Flower ClientApp
    app = ClientApp(
        client_fn,
    )
    return app
