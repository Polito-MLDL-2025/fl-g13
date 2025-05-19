from collections import defaultdict
from enum import Enum

import torch
import tqdm
from fl_g13.fl_pytorch.client import CustomNumpyClient

class TrainingPhase(Enum):
    WARMUP = "WARMUP"
    MASK_CALIBRATION = "MASK_CALIBRATION"
    TRAINING = "TRAINING"

class FullyCentralizedMaskedClient(CustomNumpyClient):
    def __init__(
        self,
        client_state,
        local_epochs,
        trainloader, 
        valloader,
        model,
        criterion,
        optimizer, 
        scheduler=None,
        device=None,
        *args, 
        **kwargs
    ):
        super().__init__(
            client_state=client_state,
            local_epochs=local_epochs,
            trainloader=trainloader, 
            valloader=valloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer, 
            scheduler=scheduler,
            device=device,
            *args, 
            **kwargs
        )

        self.phase = TrainingPhase.WARMUP
        self.mask = [torch.ones_like(p, device=p.device) for p in model.parameters()]
        self.set_mask(self.mask)

    def fit(self, parameters, config):
        print("Entered client fit()")

        # Parse phase and optional mask from server config
        if "phase" not in config:
            print("Warning: 'phase' not found in config, defaulting to WARMUP")
            phase = TrainingPhase.WARMUP
        else:
            phase = TrainingPhase(config["phase"])  # Ensure Enum type
        self.phase = phase

        # Convert received mask lists back to tensors on the correct device
        if "mask" not in config:
            print("Warning: 'mask' not found in config, using default ones mask")
            mask = [torch.ones_like(p, device=self.device) for p in self.model.parameters()]
        else:
            received_mask = config["mask"]
            mask = [torch.tensor(m, device=self.device) for m in received_mask]
        self.set_mask(mask)

        # Phase-specific behavior
        if phase == TrainingPhase.MASK_CALIBRATION:
            # Compute fisher scores
            fisher_scores = self._masked_fisher_score(
                dataloader=self.trainloader,
                model=self.model,
                current_mask=self.mask,
            )
            # Send fisher scores back
            return fisher_scores, 0, None
        elif phase == TrainingPhase.TRAINING or phase == TrainingPhase.WARMUP:
            updated_weights, datalen, results = super().fit(parameters, config)
        else:
            raise ValueError(f"Unsupported client phase: {phase}")
        
        return updated_weights, datalen, results
    
    def _masked_fisher_score(
        self,
        dataloader,
        model,
        current_mask,
        verbose = 1,
        loss_fn = torch.nn.CrossEntropyLoss()
    ):
        # Get the device where the model is located
        device = next(model.parameters()).device
        # Set the model to evaluation mode    
        model.eval()
        
        # Set the for loop iterator according to the verbose flag
        if verbose == 1:
            # Default, use tqdm with progress bar
            batch_iterator = tqdm(dataloader, desc = 'Fisher Score', unit = 'batch')
        else:
            # No progress bar
            batch_iterator = dataloader
            
        # Initialize variables
        fisher_scores = defaultdict(lambda: 0)
        total_batches = len(dataloader)
        
        for name, param in model.named_parameters():
            if name not in current_mask:
                current_mask[name] = torch.ones_like(param.data).to(device)
            # Move mask to the correct device
            current_mask[name] = current_mask[name].to(device)
        
        for batch_idx, (X, y) in enumerate(batch_iterator):
            X, y = X.to(device), y.to(device)

            model.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    masked_grad = param.grad.detach() * current_mask[name].to(param.grad.device)
                    fisher_scores[name] += (masked_grad ** 2)
                    
            # Verbose == 2 print progress every 10 batches
            if verbose == 2 and (batch_idx + 1) % 10 == 0:
                print(f"  ↳ Batch {batch_idx + 1}/{total_batches}")
                
        # Average over number of batches
        for name in fisher_scores:
            fisher_scores[name] /= total_batches
            
        return fisher_scores
