from fl_g13.fl_pytorch.client import CustomNumpyClient
import torch
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.fl_pytorch.model import get_experiment_setting
import json
import numpy as np
from fl_g13.modeling import train
from flwr.common import RecordDict, ConfigRecord
from fl_g13.editing.masking import uncompress_mask_sparse

class WarmUpHeadTalosClient(CustomNumpyClient):

    def __init__(
            self,
            client_state: RecordDict,
            local_epochs,
            trainloader,
            valloader,
            model,
            criterion,
            optimizer,
            scheduler=None,
            device=None,
            model_editing=False,
            sparsity=0.8,
            mask_type='global',
            is_save_weights_to_state=False,
            verbose=0,
            mask_calibration_round=1,
            warm_up_rounds=4,
            *args, 
            **kwargs
    ):
        super().__init__(
            client_state=client_state,
            local_epochs=local_epochs,
            trainloader=trainloader, 
            valloader=valloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer, 
            scheduler=scheduler,
            device=device,
            verbose=verbose,
            mask_calibration_round=mask_calibration_round,
            *args, 
            **kwargs
        )
        self.client_state = client_state
        self.mask_list = None
        self.model_editing = model_editing
        self.is_save_weights_to_state = is_save_weights_to_state
        self.verbose = verbose
        self.sparsity = sparsity
        self.mask_type = mask_type
        self.first_time = True
        self.warm_up_rounds = warm_up_rounds

        self.model.to(self.device)
        
    
    def _warm_up_classification_head(self, params):
        print(f"Fine-tuning classification head")
        warm_up_dino_config = {
            "dropout_rate": 0.0,
            "head_hidden_size": 512,
            "head_layers": 3,
            "unfreeze_blocks": 0,
        }
        (
            classification_head, 
            optimizer, 
            criterion, 
            device, 
            scheduler,
        ) = get_experiment_setting(
            model_editing=False, 
            model_config=warm_up_dino_config,
        )
        set_weights(model=classification_head, parameters=params)
        accuracy = 0
        epoch = 0
        while accuracy < 0.6 and epoch < 16:
            _, _, all_training_accuracies, _ = train(
                checkpoint_dir=None,
                name=None,
                start_epoch=1,
                num_epochs=1,
                save_every=None,
                backup_every=None,
                train_dataloader=self.trainloader,
                val_dataloader=None,
                model=classification_head,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                eval_every=None,
                verbose=self.verbose
            )
            accuracy = all_training_accuracies[-1]
            epoch += 1
        
        warm_up_head_params = get_weights(classification_head)
        set_weights(self.model, warm_up_head_params)
    
    def fit(self, parameters, config):

        num_server_round = config.get("server_round", 0)

        if num_server_round < self.warm_up_rounds:
            self._warm_up_classification_head(params=parameters)
        
        else:
            print("finished warm up")
            if "participation" not in self.client_state:
                self.client_state["participation"] = ConfigRecord()

            participation_record = self.client_state["participation"]

            first_time = not participation_record.get("has_participated", False)

            if first_time:
                print(f"First time participating in training")
                participation_record["has_participated"] = True
                self._warm_up_classification_head(params=parameters)
                self._compute_mask(sparsity=self.sparsity, mask_type=self.mask_type)
                self._save_mask_to_state()
            else:
                self._load_mask_from_state()

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
        
        # Save weights from global models
        flatten_global_weights = np.concatenate([p.flatten() for p in parameters])

        # fine tuned weights
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

        return (
            updated_weights,
            len(self.trainloader.dataset),
            results,
            # if you have more complex metrics you have to serialize them with json since Metrics value allow only Scalar
        )