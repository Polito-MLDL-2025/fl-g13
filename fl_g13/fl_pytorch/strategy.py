import json
from logging import INFO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    logger,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling import save

WANDB_TEST_PROJECT_NAME = "CIFAR100_FL_experiment"


class CustomFedAvg(FedAvg):
    """
    Custom federated averaging strategy that extends FedAvg to include:
    - Regular server-side model checkpointing.
    - Logging of training and evaluation metrics to Weights & Biases.
    - Calculation and logging of client parameter drift.
    - Periodic centralized evaluation on the server.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str,
        model: torch.nn.Module,
        start_epoch: int = 1,
        save_every: int = 1,
        save_with_model_dir: bool = False,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        evaluate_each: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the CustomFedAvg strategy.

        Args:
            checkpoint_dir (str): Directory to save model checkpoints.
            prefix (str): A prefix for saved model checkpoint files.
            model (torch.nn.Module): The PyTorch model to be trained and saved.
            start_epoch (int, optional): The starting epoch number. Defaults to 1.
            save_every (int, optional): Frequency of saving checkpoints (in epochs). Defaults to 1.
            save_with_model_dir (bool, optional): If True, saves checkpoints in a subdirectory named after the model. Defaults to False.
            use_wandb (bool, optional): If True, enables W&B logging. Defaults to False.
            wandb_config (dict, optional): Configuration for W&B initialization. Defaults to None.
            evaluate_each (int, optional): Frequency of performing centralized evaluation. Defaults to 1.
        """
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.save_with_model_dir = save_with_model_dir
        self.model = model
        self.start_epoch = start_epoch
        self.save_every = save_every
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config if wandb_config is not None else {}
        self.evaluate_each = evaluate_each
        if use_wandb:
            self._init_wandb_project()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates client training results, calculates custom metrics, and saves checkpoints.

        This method extends the base `aggregate_fit` to compute the average training
        accuracy and loss across all successful clients. It also calculates the
        average L2 norm of the difference between client parameters and the global
        model parameters (drift), logs metrics to W&B, and saves the aggregated
        model at specified intervals.

        Args:
            server_round (int): The current round of federated learning.
            results (list[tuple[ClientProxy, FitRes]]): List of successful client results.
            failures (list[BaseException]): List of exceptions from failed clients.

        Returns:
            tuple[Parameters | None, dict]: The aggregated parameters and a dictionary of metrics.
        """
        # Perform standard federated averaging
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # If no clients succeeded, aggregated_params will be None.
        if aggregated_params is None:
            logger.log(
                INFO,
                f"[Round {server_round}] No aggregated parameters (possibly all clients failed).",
            )
            return None, {}

        # Calculate decentralized (client-side) training metrics and drift
        total_accuracy = 0.0
        total_loss = 0.0
        total_examples = 0
        total_drift = 0.0
        global_params_flat = np.concatenate(
            [p.flatten() for p in get_weights(self.model)]
        )

        for _, fit_res in results:
            num_examples = fit_res.num_examples
            client_metrics = fit_res.metrics

            # Calculate drift (L2 norm between client and global model parameters)
            client_params_flat = np.concatenate(
                [p.flatten() for p in parameters_to_ndarrays(fit_res.parameters)]
            )
            drift = np.linalg.norm(client_params_flat - global_params_flat)
            total_drift += drift

            # Accumulate weighted metrics
            total_accuracy += (
                json.loads(client_metrics["training_accuracies"])[-1] * num_examples
            )
            total_loss += (
                json.loads(client_metrics["training_losses"])[-1] * num_examples
            )
            total_examples += num_examples

        # Avoid division by zero
        avg_accuracy = total_accuracy / total_examples if total_examples > 0 else 0.0
        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        avg_drift = total_drift / total_examples if total_examples > 0 else 0.0

        # Log decentralized metrics to W&B
        self.wandb_log(
            server_round=server_round,
            results_dict={
                "decentralized_avg_train_loss": avg_loss,
                "decentralized_avg_train_accuracy": avg_accuracy,
            },
        )

        # Calculate relative drift
        param_arrays = parameters_to_ndarrays(aggregated_params)
        flat_params = np.concatenate([arr.flatten() for arr in param_arrays])
        global_l2_norm = np.linalg.norm(flat_params)
        relative_drift = avg_drift / (global_l2_norm + 1e-8)

        # Update metrics dictionary and log drift
        aggregated_metrics["avg_drift"] = avg_drift
        aggregated_metrics["relative_drift"] = relative_drift
        logger.log(
            INFO,
            f"[Round {server_round}] Avg Drift: {avg_drift:.4f} | Relative Drift: {relative_drift:.4f}",
        )

        # Save model checkpoint periodically
        epoch = self.start_epoch + server_round - 1
        if self.checkpoint_dir and self.save_every and epoch % self.save_every == 0:
            logger.log(
                INFO, f"[Round {server_round}] Saving aggregated model at epoch {epoch}..."
            )
            set_weights(self.model, param_arrays)
            save(
                checkpoint_dir=self.checkpoint_dir,
                model=self.model,
                prefix=f"fl_{self.prefix}",
                epoch=epoch,
                with_model_dir=self.save_with_model_dir,
            )

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates evaluation results and logs them to W&B.

        Args:
            server_round (int): The current round of federated learning.
            results (list[tuple[ClientProxy, EvaluateRes]]): Successful evaluation results.
            failures (list[BaseException]): Exceptions from failed clients.

        Returns:
            tuple[float | None, dict]: The aggregated loss and a dictionary of metrics.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Log decentralized (client-side) evaluation metrics
        self.wandb_log(
            server_round=server_round,
            results_dict={"decentralized_avg_eval_loss": loss, **metrics},
        )

        return loss, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Performs centralized evaluation on the server.

        This evaluation is triggered periodically based on the `evaluate_each`
        parameter. The results are logged to the console and W&B.

        Args:
            server_round (int): The current round of federated learning.
            parameters (Parameters): The current global model parameters.

        Returns:
            tuple[float, dict[str, Scalar]] | None: Loss and metrics if evaluated, else None.
        """
        epoch = self.start_epoch + server_round - 1
        if epoch % self.evaluate_each == 0:
            loss, metrics = super().evaluate(server_round, parameters)
            logger.log(
                INFO,
                f"[Round {server_round}] Centralized Evaluation - Loss: {loss:.4f}, Metrics: {metrics}",
            )
            self.wandb_log(
                server_round=server_round,
                results_dict={"centralized_eval_loss": loss, **metrics},
            )
            return loss, metrics
        # Return empty dict if not evaluating to avoid issues with Flower
        return None, {}

    def _init_wandb_project(self) -> None:
        """
        Initializes the Weights & Biases project and run.

        It attempts to resume a previous run by reading a run ID from
        `wandb_run_id.txt`. If the file doesn't exist, a new run is created
        and its ID is saved.
        """
        run_id = self.wandb_config.get("run_id")
        if not run_id:
            run_id_path = Path.cwd() / "wandb_run_id.txt"
            if run_id_path.exists():
                run_id = run_id_path.read_text().strip()
            else:
                run_id = wandb.util.generate_id()
                run_id_path.write_text(run_id)

        # Set default values for W&B init if not provided
        name = self.wandb_config.get(
            "name",
            f"{self.model.__class__.__name__}-{self.wandb_config.get('partition_type', 'default')}",
        )
        project_name = self.wandb_config.get("project_name", WANDB_TEST_PROJECT_NAME)
        resume = self.wandb_config.get("resume", "allow")

        wandb.init(
            project=project_name,
            name=name,
            config=self.wandb_config,
            id=run_id,
            resume=resume,
        )

    def wandb_log(self, server_round: int, results_dict: Dict[str, Any]) -> None:
        """
        Logs a dictionary of results to Weights & Biases if enabled.

        Args:
            server_round (int): The current server round, used to calculate the step.
            results_dict (dict): A dictionary of metrics to log.
        """
        if self.use_wandb:
            # Calculate the current epoch for logging
            current_epoch = self.start_epoch + server_round - 1
            wandb.log(results_dict, step=current_epoch)