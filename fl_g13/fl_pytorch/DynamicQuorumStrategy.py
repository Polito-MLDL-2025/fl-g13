import numpy as np

from logging import INFO

from flwr.common.typing import FitIns
from flwr.common import logger

from fl_g13.editing import compress_mask_sparse, uncompress_mask_sparse
from fl_g13.fl_pytorch.strategy import CustomFedAvg

class DynamicQuorum(CustomFedAvg):
    def __init__(
        self,
        mask_sum,
        num_total_clients,
        quorum_update_frequency=10,
        initial_quorum=1,
        quorum_increment=10,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # --- Quorum Parameters ---
        self.num_total_clients = num_total_clients
        self.quorum_update_frequency = quorum_update_frequency
        self.initial_quorum = initial_quorum
        self.quorum_increment = quorum_increment
        
        # --- State Tracking ---
        self.current_quorum = self.initial_quorum
        # The mask sum is now computed ONCE at initialization
        self.mask_sum = mask_sum
        self.global_mask = None # Will be generated in the first configure_fit call

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Dynamically adjusts the quorum and creates the global mask before sending it to clients.
        """
        updated_quorum = server_round == 1
        
        # --- DYNAMIC QUORUM LOGIC ---
        # 1. Check if it's time to update the quorum (but not in the very first round)
        if server_round > 1 and (server_round - 1) % self.quorum_update_frequency == 0:
            self.current_quorum = min(self.num_total_clients, self.current_quorum + self.quorum_increment)
            logger.log(INFO, f"[Round {server_round}] Quorum updated to: {self.current_quorum}")
            updated_quorum = True

        # 2. Generate the global mask using the current quorum on the static mask_sum
        if self.mask_sum is not None and updated_quorum:
            self.global_mask = [(layer_sum >= self.current_quorum).float() for layer_sum in self.mask_sum]
            
            # Optional: Log the new sparsity
            # Log initial state
            total_params = sum(np.prod(layer.shape) for layer in self.global_mask)
            total_non_zero = sum(layer.cpu().numpy().nonzero()[0].size for layer in self.global_mask)
            sparsity = 1.0 - (total_non_zero / total_params)
            logger.log(INFO, f"[Round {server_round}] Generated global mask with sparsity: {sparsity:.4f}")

        # 3. Prepare the configuration to be sent to the clients
        config = {}
        if self.global_mask is not None:
            config["global_mask"] = compress_mask_sparse(self.global_mask)
        
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients)
        return [(client, fit_ins) for client in clients]
    
    # aggregate_fit it unchanged