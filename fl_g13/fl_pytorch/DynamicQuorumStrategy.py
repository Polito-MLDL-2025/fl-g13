import numpy as np

from logging import INFO

from flwr.common.typing import FitIns
from flwr.common import logger

from fl_g13.editing import compress_mask_sparse, uncompress_mask_sparse
from fl_g13.fl_pytorch.strategy import CustomFedAvg

def _compute_sparsity_given_quorum(mask, q):
    assert 0 < q <= 100
    
    global_mask = [(layer_sum >= q).float() for layer_sum in mask]
    total_params = sum(np.prod(layer.shape) for layer in global_mask)
    total_non_zero = sum(layer.cpu().numpy().nonzero()[0].size for layer in global_mask)
    return 1.0 - (total_non_zero / total_params)

def _set_initial_quorum(mask_sum, total_clients = 100, target_sparsity = 0.7):
    if _compute_sparsity_given_quorum(mask_sum, total_clients) < target_sparsity: # Check if it's possible to achieve target sparsity
        return -1
    
    for q in range(1, total_clients + 1):
        sparsity = _compute_sparsity_given_quorum(mask_sum, q)
        if sparsity >= target_sparsity:
            return q
    
    # If here, than there's a least a value that satisfy the requirement and must be the last one
    #   since the for loop did not return prematurely
    return total_clients # fallback

class DynamicQuorum(CustomFedAvg):
    def __init__(
        self,
        mask_sum,
        num_total_clients,
        quorum_update_frequency=10,
        initial_quorum=1,
        quorum_increment=10,
        # --- ADAPTIVE MODE ---
        adaptive_quorum: bool = False,
        initial_target_sparsity: float = 0.7,
        drift_threshold: float = 0.5,
        quorum_patience: int = 2,
        force_quorum_update: int = 15,
        *args, **kwargs
    ):
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
        self.global_mask = None
        self.sparsity = 0.0
        self.last_quorum_update_round = 1 # Track when the last update happened
        
        # --- Set initial quorum ---
        # Only in ADAPTIVE mode
        if self.adaptive_quorum:
            self.current_quorum = _set_initial_quorum(self.mask_sum, self.num_total_clients, target_sparsity = initial_target_sparsity)
            # if cannot reach target sparsity, return to LINEAR mode
            if self.current_quorum < 0:
                self.adaptive_quorum = False
                self.current_quorum = initial_quorum            

        if self.adaptive_quorum:
            logger.log(INFO, f"[DQ] ADAPTIVE mode enabled. Quorum: {self.current_quorum}; Drift threshold: {self.drift_threshold}")
        else:
            logger.log(INFO, f"[DQ] LINEAR mode enabled. Quorum: {self.current_quorum}; Update frequency: {self.quorum_update_frequency} rounds.")

    def configure_fit(self, server_round, parameters, client_manager):
        updated_quorum = False
        
        # 1. Use LINEAR update logic if adaptive mode is OFF
        if not self.adaptive_quorum:
            if server_round > 1 and server_round % self.quorum_update_frequency == 0:
                self.current_quorum = min(self.num_total_clients, self.current_quorum + self.quorum_increment)
                logger.log(INFO, f"[Round {server_round} DQ-LINEAR]. New Quorum: {self.current_quorum}")
                updated_quorum = True
        
        # Flag that is set by aggregate_fit in adaptive mode
        if self.last_quorum_update_round == server_round - 1:
            updated_quorum = True

        # Generate the global mask if it's the first round or if the quorum was updated
        if server_round == 1 or updated_quorum:
            if self.mask_sum is not None:
                self.global_mask = [(layer_sum >= self.current_quorum).float() for layer_sum in self.mask_sum]
            
                # Log the new sparsity
                self.sparsity = _compute_sparsity_given_quorum(self.mask_sum, self.current_quorum)
                logger.log(INFO, f"[Round {server_round} DQ] Generated global mask with sparsity: {self.sparsity:.4f}")

        config = {}
        if self.global_mask is not None:
            # Move to CPU before serialization
            cpu_mask = [layer.cpu() for layer in self.global_mask]
            config["global_mask"] = compress_mask_sparse(cpu_mask)
        
        # Log to wandb
        if self.use_wandb:
            wandb_quorum_stats = { 'quorum': self.current_quorum, 'mask_sparsity': self.sparsity }
            self.wandb_log(server_round = server_round, results_dict = wandb_quorum_stats)
        
        # Send to clients
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # --- ADAPTIVE QUORUM UPDATE LOGIC ---
        if self.adaptive_quorum and aggregated_params is not None:
            # Average Dirft is used for the quorum update
            avg_drift = aggregated_metrics.get("avg_drift", float('inf'))
            
            # Check conditions: 
            ### Force quorum update
            if (server_round + 1 - self.last_quorum_update_round) % self.force_quorum_update == 0:
                self.current_quorum = min(self.num_total_clients, self.current_quorum + self.quorum_increment)
                self.last_quorum_update_round = server_round
                logger.log(INFO, f"[Round {server_round} DQ-ADAPTIVE] New Quorum: {self.current_quorum} (Forcing an update)")
            ### Stable clients: Drift is below threshold
            if avg_drift < self.drift_threshold and (server_round - self.last_quorum_update_round) >= self.quorum_patience:
                self.current_quorum = min(self.num_total_clients, self.current_quorum + self.quorum_increment)
                self.last_quorum_update_round = server_round
                logger.log(INFO, f"[Round {server_round} DQ-ADAPTIVE] New Quorum: {self.current_quorum} (Drift ({avg_drift}) < Threshold ({self.drift_threshold}))")
        
        return aggregated_params, aggregated_metrics