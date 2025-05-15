from collections import defaultdict
from enum import Enum

import torch
from fl_g13.editing.masking import create_gradiend_mask
from fl_g13.fl_pytorch.strategy import CustomFedAvg

class TrainingPhase(Enum):
    WARMUP = "WARMUP"
    MASK_CALIBRATION = "MASK_CALIBRATION"
    TRAINING = "TRAINING"

#! TODO: For some reason, if this class is runned, the clients will not run the fit() function
#!       I suspect it is due to the use of configure_fit() 
class FullyCentralizedMaskedFedAvg(CustomFedAvg):
    def __init__(
        self,
        checkpoint_dir,
        prefix,
        model,
        start_epoch=1,
        save_every=1,
        save_with_model_dir=False,
        use_wandb = False,
        wandb_config=None,
        sparsity=0.8,
        num_warmup_rounds=1,
        *args,
        **kwargs
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            prefix=prefix,
            model=model,
            start_epoch=start_epoch,
            save_every=save_every,
            save_with_model_dir=save_with_model_dir,
            use_wandb=use_wandb,
            wandb_config=wandb_config,
            *args,
            **kwargs
        )

        self.num_warmup_rounds = num_warmup_rounds
        self.phase = TrainingPhase.WARMUP
        self.desired_sparsity = sparsity
        self.sparsity = sparsity
        self.mask = [torch.ones_like(p, device=p.device) for p in model.parameters()]
    
    def configure_fit(self, server_round, parameters, client_manager):
        print("Entered server configure_fit()")
        
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        print(f"Number of client instructions: {len(client_instructions)}")
        
        # Convert mask tensors to CPU and then to lists for serialization
        serializable_mask = [tensor.cpu().tolist() for tensor in self.mask]
        
        for client_id, fit_ins in client_instructions:
            fit_ins.config["phase"] = self.phase.value
            fit_ins.config["mask"] = serializable_mask
        
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        print("Entered server aggregate_fit()")

        if self.phase == TrainingPhase.MASK_CALIBRATION:
            if not results:
                print("No results received during mask calibration")
                return None, {}
            # Collect fisher scores
            fisher_scores = defaultdict(lambda: 0)
            for _, fit_res in results:
                client_fisher_scores = fit_res.metrics["fisher_scores"]
                for name, score in client_fisher_scores.items():
                    fisher_scores[name] += score

            # Calculate mask based on fisher scores
            self.mask = create_gradiend_mask(
                class_score=fisher_scores,
                sparsity=self.sparsity,
                mask_type='global'
            )
            
            # Calculate actual sparsity achieved
            total_params = sum(m.numel() for m in self.mask)
            zero_params = sum((m == 0).sum().item() for m in self.mask)
            achieved_sparsity = zero_params / total_params
            print(f"Achieved sparsity: {achieved_sparsity}")

            if achieved_sparsity < self.desired_sparsity:
                print("Target sparsity not reached, continuing calibration")
                self.phase = TrainingPhase.MASK_CALIBRATION
            else:
                print("Target sparsity reached, switching to training phase")
                self.phase = TrainingPhase.TRAINING
            
            # Return empty params/metrics since this is calibration phase
            return None, {}

        elif self.phase == TrainingPhase.TRAINING or self.phase == TrainingPhase.WARMUP:
            aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            
            if server_round >= self.num_warmup_rounds and self.phase == TrainingPhase.WARMUP:
                print(f"Warmup complete after {self.num_warmup_rounds} rounds, switching to mask calibration")
                self.phase = TrainingPhase.MASK_CALIBRATION
            
            return aggregated_params, aggregated_metrics
        else:
            raise ValueError(f"Unsupported server phase: {self.phase}")