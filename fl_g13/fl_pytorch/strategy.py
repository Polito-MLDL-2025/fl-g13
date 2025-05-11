import copy
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from flwr.common import logger, parameters_to_ndarrays, ndarrays_to_parameters #! logger doesnt work
from flwr.server.strategy import FedAvg
import wandb

from fl_g13.editing.masking import create_gradiend_mask
from fl_g13.fl_pytorch.task import set_weights
from fl_g13.modeling import save

WANDB_PROJECT_NAME = "CIFAR100_FL_experiment"

# *** -------- AGGREGATION SERVER STRATEGY -------- *** #

class CustomFedAvg(FedAvg):
    def __init__(
        self,
        checkpoint_dir,
        model,
        start_epoch=1,
        save_every=1,
        #save_best_model = True, #!! Removed
        use_wandb = False,
        wandb_config=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.start_epoch = start_epoch
        self.save_every = save_every
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config
        if use_wandb:
            self._init_wandb_project()

    # -------- AGGREGATION -------- #

    def aggregate_fit(self, server_round, results, failures):
        
        # Retrieve results from standard FedAvg using masked results
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # If no aggregate params are available then no client trained succesffuly, warn and skip
        if aggregated_params is None:
            #logger.warning(f"[Round {server_round}] No aggregated parameters (possibly all clients failed).")
            print(f"[Round {server_round}] No aggregated parameters (possibly all clients failed).")
            return None, {}

        # Convert parameters to NumPy arrays for analysis
        param_arrays = parameters_to_ndarrays(aggregated_params)
        flat_params = np.concatenate([arr.flatten() for arr in param_arrays])
        global_l2_norm = np.linalg.norm(flat_params)

        # Compute and log drift if available
        if "avg_drift" in aggregated_metrics:
            avg_drift = aggregated_metrics["avg_drift"]
            relative_drift = avg_drift / (global_l2_norm + 1e-8)
            #logger.info(f"[Round {server_round}] Avg Drift: {avg_drift:.4f} | Relative Drift: {relative_drift:.4f}")
            print(f"[Round {server_round}] Avg Drift: {avg_drift:.4f} | Relative Drift: {relative_drift:.4f}")

        # Optionally save model checkpoint
        epoch = self.start_epoch + server_round - 1
        if self.checkpoint_dir and self.save_every and epoch % self.save_every == 0:
            #logger.info(f"[Round {server_round}] Saving aggregated model at epoch {epoch}...")
            print(f"[Round {server_round}] Saving aggregated model at epoch {epoch}...")
            set_weights(self.model, param_arrays)
            save(
                checkpoint_dir=self.checkpoint_dir,
                model=self.model,
                prefix="FL",
                epoch=epoch,
                with_model_dir=False
            )

        return aggregated_params, aggregated_metrics

    # Wrap aggregate evaluate of FedAvg (does nothing)
    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)

    # Wrap aggregate evaluate of FedAvg and prints
    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        #logger.info(f"[Round {server_round}] Centralized Evaluation - Loss: {loss:.4f}, Metrics: {metrics}")
        print(f"[Round {server_round}] Centralized Evaluation - Loss: {loss:.4f}, Metrics: {metrics}")
        self.wandb_log(
            server_round=server_round,
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    # -------- WANDB UTILITIES -------- #

    def _init_wandb_project(self):
        
        # save or read run_id to be able to resume the run
        run_id_path = Path.cwd() / "wandb_run_id.txt"
        if os.path.exists(run_id_path):
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()
            with open(run_id_path, "w") as f:
                f.write(run_id)
        
        # init W&B
        wandb.init(
            project=WANDB_PROJECT_NAME, 
            name=f"{self.model.__class__.__name__}-{self.wandb_config['partition_type']}",
            config=self.wandb_config,
            id=run_id,
            resume="allow",
        )

    ##! Same as store_results_and_log, but skipping the local storage (as it was not done)
    def wandb_log(self, server_round: int, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=self.start_epoch + server_round - 1)

    #! Removed _store_results and _update_best_acc
