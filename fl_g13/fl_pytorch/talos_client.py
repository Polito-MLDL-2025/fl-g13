from fl_g13.fl_pytorch.base_client import FlowerClient
import torch
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.fl_pytorch.model import get_experiment_setting
import json
import numpy as np
from fl_g13.modeling import train
from flwr.common import RecordDict, ConfigRecord

class TalosClient(FlowerClient):

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

        self.model.to(self.device)

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
    
    def _catch_up_classification_head(self):
        print(f"Fine-tuning classification head")
        lw_dino_config = {
            "dropout_rate": 0.0,
            "head_hidden_size": 512,
            "head_layers": 3,
            "unfreeze_blocks": 0,
        }
        (
            model, 
            optimizer, 
            criterion, 
            device, 
            scheduler,
        ) = get_experiment_setting(
            model_editing=False, 
            model_config=lw_dino_config,
        )

        accuracy = 0
        epoch = 0
        while accuracy < 0.6 and epoch < 16:
            _, _, accuracy, _ = train(
                checkpoint_dir=None,
                name=None,
                start_epoch=1,
                num_epochs=1,
                save_every=None,
                backup_every=None,
                train_dataloader=self.trainloader,
                val_dataloader=None,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                eval_every=None,
                verbose=self.verbose
            )
            epoch += 1
        
        fine_tuned_head_params = get_weights(model)
        set_weights(self.model, fine_tuned_head_params)
    
    def fit(self, parameters, config):

        if "participation" not in self.client_state:
            self.client_state["participation"] = ConfigRecord()

        participation_record = self.client_state["participation"]

        first_time = not participation_record.get("has_participated", False)

        if first_time:
            print(f"First time participating in training")
            participation_record["has_participated"] = True
            self._catch_up_classification_head()
            self._compute_mask(sparsity=self.sparsity, mask_type=self.mask_type)

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

        fit_params = self._compute_task_vector(updated_weights, parameters)

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