import numpy as np
import torch.nn as nn
from flwr.common import logger, parameters_to_ndarrays #! logger doesnt work
from flwr.server.strategy import FedAvg

from fl_g13.fl_pytorch.task import set_weights
from fl_g13.modeling import save

# *** -------- AGGREGATION SERVER STRATEGY -------- *** #

class MaskedFedAvg(FedAvg):
    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir,
        start_epoch= 1,
        save_every= 1,
        save_best_model: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = start_epoch
        self.save_every = save_every
        self.save_best_model = save_best_model
        self.results = {}

    def aggregate_fit(self, server_round, results, failures):

        #TODO Server Masking (aggregation of masks, or pure centralized aggregation)

        # Retrive results from standard FedAvg
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
        return loss, metrics
