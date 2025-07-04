import numpy as np
from logging import INFO
from typing import Any, Dict, List, Optional, Tuple

import torch
from flwr.common import FitIns, FitRes, Parameters, Scalar, logger
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl_g13.fl_pytorch.strategy import CustomFedAvg
from fl_g13.editing import compress_mask_sparse

def _compute_sparsity_given_quorum(
    mask: List[torch.Tensor], q: int, max_clients: int = 100
) -> float:
    """
    Calculates the sparsity of a global mask that would be generated with a given quorum.

    Args:
        mask (List[torch.Tensor]): A list of tensors representing the sum of client masks.
        q (int): The quorum threshold.
        max_clients (int, optional): The maximum number of clients. Defaults to 100.

    Returns:
        float: The calculated sparsity, as a value between 0.0 and 1.0.
    """
    assert 0 < q <= max_clients, f"Quorum must be between 1 and {max_clients}"

    # A parameter is included in the global mask if at least `q` clients have it in their masks
    global_mask = [(layer_sum >= q).float() for layer_sum in mask]
    total_params = sum(np.prod(layer.shape) for layer in global_mask)
    total_non_zero = sum(layer.cpu().numpy().nonzero()[0].size for layer in global_mask)

    # Sparsity is the fraction of zeroed-out parameters
    return 1.0 - (total_non_zero / total_params)

def _set_initial_quorum(
    mask_sum: List[torch.Tensor],
    total_clients: int = 100,
    target_sparsity: float = 0.7,
) -> int:
    """
    Determines the initial quorum required to meet a target sparsity level.

    It iterates through possible quorum values to find the smallest one that achieves
    the desired sparsity.

    Args:
        mask_sum (List[torch.Tensor]): The sum of all client masks.
        total_clients (int, optional): The total number of clients. Defaults to 100.
        target_sparsity (float, optional): The desired sparsity level. Defaults to 0.7.

    Returns:
        int: The calculated initial quorum, or -1 if the target sparsity is unachievable.
    """
    # Check if the target sparsity is even possible
    if (
        _compute_sparsity_given_quorum(mask_sum, total_clients, total_clients)
        < target_sparsity
    ):
        return -1

    # Find the smallest quorum that meets the target sparsity
    for q in range(1, total_clients + 1):
        sparsity = _compute_sparsity_given_quorum(mask_sum, q, total_clients)
        if sparsity >= target_sparsity:
            return q

    # Fallback if no suitable quorum is found (should not be reached if the initial check passes)
    return total_clients

class DynamicQuorum(CustomFedAvg):
    """
    A federated learning strategy with a dynamic client quorum.

    This strategy extends `CustomFedAvg` to adjust the client quorum required for
    parameter aggregation. The quorum can be updated based on a fixed linear
    schedule or an adaptive mechanism that responds to model drift.
    """

    def __init__(
        self,
        mask_sum: List[torch.Tensor],
        num_total_clients: int,
        quorum_update_frequency: int = 10,
        initial_quorum: int = 1,
        quorum_increment: int = 10,
        # --- ADAPTIVE MODE ---
        adaptive_quorum: bool = False,
        initial_target_sparsity: float = 0.7,
        drift_threshold: float = 0.5,
        quorum_patience: int = 2,
        force_quorum_update: int = 15,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the DynamicQuorum strategy.

        Args:
            mask_sum (List[torch.Tensor]): The sum of masks from all clients.
            num_total_clients (int): The total number of clients.
            quorum_update_frequency (int, optional): The frequency (in rounds) for linear quorum updates. Defaults to 10.
            initial_quorum (int, optional): The starting quorum value. Defaults to 1.
            quorum_increment (int, optional): The amount to increase the quorum by. Defaults to 10.
            adaptive_quorum (bool, optional): Enables adaptive quorum updates based on drift. Defaults to False.
            initial_target_sparsity (float, optional): The target sparsity for adaptive mode. Defaults to 0.7.
            drift_threshold (float, optional): The drift threshold to trigger quorum updates in adaptive mode. Defaults to 0.5.
            quorum_patience (int, optional): The number of stable rounds to wait before increasing the quorum. Defaults to 2.
            force_quorum_update (int, optional): The number of rounds after which to force a quorum update. Defaults to 15.
        """
        super().__init__(*args, **kwargs)
        # --- Quorum Parameters ---
        self.num_total_clients = num_total_clients
        self.quorum_update_frequency = quorum_update_frequency
        self.initial_quorum = initial_quorum
        self.quorum_increment = quorum_increment

        # --- Adaptive Quorum Parameters ---
        self.adaptive_quorum = adaptive_quorum
        self.drift_threshold = drift_threshold
        self.quorum_patience = quorum_patience
        self.force_quorum_update = force_quorum_update

        # --- State Tracking ---
        self.current_quorum = initial_quorum
        self.mask_sum = mask_sum
        self.global_mask: Optional[List[torch.Tensor]] = None
        self.sparsity = 0.0
        self.last_quorum_update_round = 1  # Track when the last update happened

        # --- Set initial quorum for adaptive mode ---
        if self.adaptive_quorum:
            self.current_quorum = _set_initial_quorum(
                self.mask_sum, self.num_total_clients, initial_target_sparsity
            )
            # If target sparsity is unreachable, revert to linear mode
            if self.current_quorum < 0:
                self.adaptive_quorum = False
                self.current_quorum = initial_quorum

        if self.adaptive_quorum:
            logger.log(
                INFO,
                f"[DQ] ADAPTIVE mode enabled. Quorum: {self.current_quorum}; Drift threshold: {self.drift_threshold}",
            )
        else:
            logger.log(
                INFO,
                f"[DQ] LINEAR mode enabled. Quorum: {self.current_quorum}; Update frequency: {self.quorum_update_frequency} rounds.",
            )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the `fit` instruction for clients.

        This method updates the quorum and generates a global mask, which is then
        sent to the clients as part of the configuration.
        """
        updated_quorum = False

        # 1. Use LINEAR update logic if adaptive mode is OFF
        if not self.adaptive_quorum:
            if server_round > 1 and server_round % self.quorum_update_frequency == 0:
                self.current_quorum = min(
                    self.num_total_clients, self.current_quorum + self.quorum_increment
                )
                logger.log(
                    INFO,
                    f"[Round {server_round} DQ-LINEAR]. New Quorum: {self.current_quorum}",
                )
                updated_quorum = True

        # Flag that is set by aggregate_fit in adaptive mode
        if self.last_quorum_update_round == server_round - 1:
            updated_quorum = True

        # Generate the global mask if it's the first round or if the quorum was updated
        if server_round == 1 or updated_quorum:
            if self.mask_sum is not None:
                self.global_mask = [
                    (layer_sum >= self.current_quorum).float()
                    for layer_sum in self.mask_sum
                ]

                # Log the new sparsity
                self.sparsity = _compute_sparsity_given_quorum(
                    self.mask_sum, self.current_quorum
                )
                logger.log(
                    INFO,
                    f"[Round {server_round} DQ] Generated global mask with sparsity: {self.sparsity:.4f}",
                )

        config = {}
        if self.global_mask is not None:
            # Move to CPU before serialization
            cpu_mask = [layer.cpu() for layer in self.global_mask]
            config["global_mask"] = compress_mask_sparse(cpu_mask)

        # Log to wandb
        if self.use_wandb:
            wandb_quorum_stats = {
                "quorum": self.current_quorum,
                "mask_sparsity": self.sparsity,
            }
            self.wandb_log(server_round=server_round, results_dict=wandb_quorum_stats)

        # Send to clients
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client `fit` results and update the quorum in adaptive mode.
        """
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # --- ADAPTIVE QUORUM UPDATE LOGIC ---
        if self.adaptive_quorum and aggregated_params is not None:
            # Average Drift is used for the quorum update
            avg_drift = aggregated_metrics.get("avg_drift", float("inf"))

            # Force quorum update after a fixed number of rounds
            if (server_round + 1 - self.last_quorum_update_round) % self.force_quorum_update == 0:
                self.current_quorum = min(
                    self.num_total_clients, self.current_quorum + self.quorum_increment
                )
                self.last_quorum_update_round = server_round
                logger.log(
                    INFO,
                    f"[Round {server_round} DQ-ADAPTIVE] New Quorum: {self.current_quorum} (Forcing an update)",
                )
            # Update quorum if drift is below threshold and enough rounds have passed
            elif (avg_drift < self.drift_threshold and 
                (server_round - self.last_quorum_update_round) >= self.quorum_patience):
                self.current_quorum = min(
                    self.num_total_clients, self.current_quorum + self.quorum_increment
                )
                self.last_quorum_update_round = server_round
                logger.log(
                    INFO,
                    f"[Round {server_round} DQ-ADAPTIVE] New Quorum: {self.current_quorum} (Drift ({avg_drift:.4f}) < Threshold ({self.drift_threshold}))",
                )

        return aggregated_params, aggregated_metrics