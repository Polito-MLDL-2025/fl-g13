import json
from logging import INFO

from flwr.common import logger

from fl_g13.fl_pytorch.client import CustomNumpyClient

from fl_g13.editing import uncompress_mask_sparse
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.train import train

class DynamicQuorumClient(CustomNumpyClient):
    def __init__(self, *args, **kwargs):
        # The __init__ method is inherited directly from CustomNumpyClient.
        # No changes are needed here. It will run the original setup logic.
        super().__init__(*args, **kwargs)

    def fit(self, parameters, config):        
        # Check if the server sent a global mask. This logic is unchanged.
        if "global_mask" in config:
            if self.verbose > 0:
                print("[Client] Received global mask from the server")
            global_mask_compressed = config["global_mask"]
            global_mask_uncompressed = uncompress_mask_sparse(global_mask_compressed)
            self.set_mask(global_mask_uncompressed)
        
        # The rest of the training logic is unchanged.
        set_weights(self.model, parameters)

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

        results = {"train_loss":  all_training_losses[-1]}
        if all_training_accuracies and all_training_losses:
            results["training_accuracies"] = json.dumps(all_training_accuracies)
            results["training_losses"] = json.dumps(all_training_losses)

        return (
            updated_weights,
            len(self.trainloader.dataset),
            results,
        )