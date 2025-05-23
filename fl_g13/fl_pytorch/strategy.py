import json
import os
from logging import INFO
from pathlib import Path
from typing import Optional, Union

import flwr
import numpy as np
import wandb
from flwr.common import logger, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_g13.fl_pytorch.task import set_weights
from fl_g13.modeling import save
from fl_g13.fl_pytorch.task import get_weights
import torch
from flwr.common import logger, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_g13.fl_pytorch.task import set_weights
from fl_g13.modeling import save

WANDB_PROJECT_NAME = "CIFAR100_FL_experiment"


# *** -------- AGGREGATION SERVER STRATEGY -------- *** #

class CustomFedAvg(FedAvg):
    def __init__(
            self,
            checkpoint_dir,
            prefix,  # !! Introduced without default value to force user to take care of this
            model,
            start_epoch=1,
            save_every=1,
            save_with_model_dir=False,  # !! Introduced
            # save_best_model = True, #!! Removed
            use_wandb=False,
            wandb_config=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.save_with_model_dir = save_with_model_dir
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

        # Compute avg accuracy from clients
        total_accuracy = 0.0
        total_loss = 0.0
        total_examples = 0
        drift = 0.0
        for _, fit_res in results:
            num_examples = fit_res.num_examples
            client_metrics = fit_res.metrics
            global_params = np.concatenate([p.flatten() for p in get_weights(self.model)])
            client_params = np.concatenate([p.flatten() for p in parameters_to_ndarrays(fit_res.parameters)])
            drift += np.linalg.norm(client_params - global_params)
            total_accuracy += json.loads(client_metrics["training_accuracies"])[-1] * num_examples
            total_loss += json.loads(client_metrics["training_losses"])[-1] * num_examples
            total_examples += num_examples

        avg_accuracy = total_accuracy / total_examples if total_examples > 0 else None
        avg_loss = total_loss / total_examples if total_examples > 0 else None
        avg_drift = drift / total_examples if total_examples > 0 else None

        self.wandb_log(
            server_round=server_round,
            results_dict={
                "decentralized_avg_train_loss": avg_loss,
                "decentralized_avg_train_accuracy": avg_accuracy},
        )

        # If no aggregate params are available then no client trained succesffuly, warn and skip
        if aggregated_params is None:
            # logger.warning(f"[Round {server_round}] No aggregated parameters (possibly all clients failed).")
            logger.log(INFO, f"[Round {server_round}] No aggregated parameters (possibly all clients failed).")
            return None, {}

        # Convert parameters to NumPy arrays for analysis
        param_arrays = parameters_to_ndarrays(aggregated_params)
        flat_params = np.concatenate([arr.flatten() for arr in param_arrays])
        global_l2_norm = np.linalg.norm(flat_params)

        
        relative_drift = avg_drift / (global_l2_norm + 1e-8)
        # logger.info(f"[Round {server_round}] Avg Drift: {avg_drift:.4f} | Relative Drift: {relative_drift:.4f}")
        logger.log(INFO,
                    f"[Round {server_round}] Avg Drift: {avg_drift:.4f} | Relative Drift: {relative_drift:.4f}")

        # Optionally save model checkpoint
        epoch = self.start_epoch + server_round - 1
        if self.checkpoint_dir and self.save_every and epoch % self.save_every == 0:
            # logger.info(f"[Round {server_round}] Saving aggregated model at epoch {epoch}...")
            logger.log(INFO, f"[Round {server_round}] Saving aggregated model at epoch {epoch}...")
            set_weights(self.model, param_arrays)
            save(
                checkpoint_dir=self.checkpoint_dir,
                model=self.model,
                prefix=f"fl_{self.prefix}",
                epoch=epoch,
                with_model_dir=self.save_with_model_dir
            )

        return aggregated_params, aggregated_metrics

    # Wrap aggregate evaluate of FedAvg (does nothing)
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.wandb_log(
            server_round=server_round,
            results_dict={
                "decentralized_avg_eval_loss": loss,
                **metrics},
        )
        return loss, metrics
    
    # Wrap aggregate evaluate of FedAvg and prints
    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        # logger.info(f"[Round {server_round}] Centralized Evaluation - Loss: {loss:.4f}, Metrics: {metrics}")
        logger.log(INFO, f"[Round {server_round}] Centralized Evaluation - Loss: {loss:.4f}, Metrics: {metrics}")
        self.wandb_log(
            server_round=server_round,
            results_dict={"centralized_eval_loss": loss, **metrics},
        )
        return loss, metrics

    # -------- WANDB UTILITIES -------- #

    def _init_wandb_project(self):

        # save or read run_id to be able to resume the run
        run_id = self.wandb_config.get("run_id") or None
        if not run_id:
            run_id_path = Path.cwd() / "wandb_run_id.txt"
            if os.path.exists(run_id_path):
                with open(run_id_path, "r") as f:
                    run_id = f.read().strip()
            else:
                run_id = wandb.util.generate_id()
                with open(run_id_path, "w") as f:
                    f.write(run_id)

        # init W&B
        name = self.wandb_config.get(
            'name') or f"{self.model.__class__.__name__}-{self.wandb_config.get('partition_type', '_')}"
        project_name = self.wandb_config.get('project_name') or WANDB_PROJECT_NAME
        resume = self.wandb_config.get('resume') or "allow"
        wandb.init(
            project=project_name,
            name=name,
            config=self.wandb_config,
            id=run_id,
            resume=resume,
        )

    ##! Same as store_results_and_log, but skipping the local storage (as it was not done)
    def wandb_log(self, server_round: int, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=self.start_epoch + server_round - 1)

    # ! Removed _store_results and _update_best_acc
    
