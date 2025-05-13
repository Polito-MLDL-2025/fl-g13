"""pytorch-example: A Flower / PyTorch app."""

import os, json
import time
from logging import INFO
from typing import Union, Optional
from pathlib import Path

import flwr
import numpy as np
import wandb
from flwr.common import logger, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import UserConfig, FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_g13.editing.masking import uncompress_mask_sparse
from fl_g13.fl_pytorch.model import get_default_model
from fl_g13.fl_pytorch.task import create_run_dir, set_weights
from fl_g13.modeling import save, save_loss_and_accuracy
from fl_g13.fl_pytorch.task import get_weights
import torch

PROJECT_NAME = "CIFAR100_FL_experiment"


class SaveModelFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool,
                 model=None, checkpoint=None,
                 save_every=1,
                 start_epoch=1,
                 save_best_model = True,
                 wandb_config=None,
                 scale_fn=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.use_wandb = use_wandb
        self.scale_fn = scale_fn

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}
        if not model:
            model = get_default_model()
        self.model = model
        self.checkpoint = checkpoint
        self.save_every = save_every
        self.start_epoch = start_epoch
        self.save_best_model = save_best_model
        self.wandb_config = wandb_config
        # Initialise W&B if set
        if use_wandb:
            self.save_path, self.run_dir = create_run_dir(run_config)
            self._init_wandb_project()

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
            project=PROJECT_NAME, 
            name=f"{self.model.__class__.__name__}-{self.wandb_config['partition_type']}",
            config=self.wandb_config,
            id=run_id,
            resume="allow",
        )

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # print(f"Saving results to {self.save_path}/results.json", self.results)
        # with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            # json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, f"ROUND {round}💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays

            ### Save best model
            if self.save_best_model:
                ndarrays = parameters_to_ndarrays(parameters)
                net = self.model
                set_weights(net, ndarrays)
                # Save the PyTorch model
                epoch = self.start_epoch + round - 1
                print(f"Saving best centralized model at epoch {epoch}...")
                filename = f"FL_{self.model.__class__.__name__}_best.pth"
                save(
                    checkpoint_dir=self.checkpoint,
                    prefix="FL",
                    model=self.model,
                    epoch=epoch,
                    filename=filename,
                    with_model_dir=False
                )

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            epoch = self.start_epoch + server_round - 1
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )
            flat_array = np.concatenate([arr.flatten() for arr in aggregated_ndarrays])
            global_norm = np.linalg.norm(flat_array)
            if "avg_drift" in aggregated_metrics:
                avg_drift = aggregated_metrics["avg_drift"]
                print(f"[Round {server_round}] Avg Client Drift: {avg_drift:.4f}")
                relative_drift = avg_drift / (global_norm + 1e-8) # Avoid division by zero
                print(f"[Round {server_round}] Relative Client Drift: {relative_drift:.4f}")
                self.store_results_and_log(
                    server_round=server_round,
                    tag="client_fit",
                    results_dict={"avg_drift": avg_drift, "relative_drift": relative_drift, **aggregated_metrics},
                )
            if self.checkpoint and self.save_every and epoch % self.save_every == 0:
                print(f"Saving centralized model epoch {epoch} aggregated_parameters...")

                set_weights(self.model, aggregated_ndarrays)
                save(
                    checkpoint_dir=self.checkpoint,
                    model=self.model,
                    prefix="FL",
                    epoch=epoch,
                    with_model_dir=False
                )

        return aggregated_parameters, aggregated_metrics
    
    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=self.start_epoch + server_round - 1)
        else:
            # Store results and save to disk
            self._store_results(
                tag=tag,
                results_dict={"round": self.start_epoch + server_round - 1, **results_dict},
            )

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        print(f"Server round {server_round} - loss: {loss}, metrics: {metrics}")

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

class ClientSideTaskArithmetic(SaveModelFedAvg):
    """
    This strategy: merge the task vectors computed by each client 
    and then apply it to the global model to teach it the specific tasks learnt by each client.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_params = get_weights(self.model)
        #global_vector = np.concatenate([gp.flatten() for gp in global_params])
        global_params = [torch.tensor(w, device=dev) for w in global_params]
        task_vecs, lambdas = [], [] 

        for client, fit_res in results:
            # one list of ndarrays per client (one ndarray per layer)
            task_vecs.append(parameters_to_ndarrays(fit_res.parameters))
            lambdas.append(self.scale_fn(client, fit_res.num_examples, server_round))  # λ_c

        # sum of the task vectors of all clients per each layer
        merged_task_vectors = [torch.zeros_like(layer_params) for layer_params in global_params]

        for lam, tau in zip(lambdas, task_vecs):
            for i, layer_params in enumerate(tau):
                merged_task_vectors[i] += lam * torch.tensor(layer_params, device=dev)


        aggregated_parameters = [
            global_layer + merged_vectors_layer 
            for global_layer, merged_vectors_layer in zip(global_params, merged_task_vectors) 
        ]

        #aggregated_parameters = ndarrays_to_parameters([aggregated_parameters.cpu().numpy()])
        aggregated_parameters = ndarrays_to_parameters([
            aggregated_parameters_layer.cpu().numpy()
            for aggregated_parameters_layer in aggregated_parameters
        ])

        aggregated_metrics = {}
        
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)


        if aggregated_parameters is not None:
            epoch = self.start_epoch + server_round - 1
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )
            if self.checkpoint and self.save_every and epoch % self.save_every == 0:
                print(f"Saving centralized model epoch {epoch} aggregated_parameters...")

                set_weights(self.model, aggregated_ndarrays)
                save(
                    checkpoint_dir=self.checkpoint,
                    model=self.model,
                    prefix="FL",
                    epoch=epoch,
                    with_model_dir=False
                )

        return aggregated_parameters, aggregated_metrics