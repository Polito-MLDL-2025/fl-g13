import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context

from fl_g13.fl_pytorch.datasets import get_transforms, load_flwr_datasets
from fl_g13.fl_pytorch.client import CustomNumpyClient
from fl_g13.fl_pytorch.FullyCentralizedMaskedClient import FullyCentralizedMaskedClient
from fl_g13.fl_pytorch.warm_up_head_talos_client import WarmUpHeadTalosClient


# *** ---------------- UTILITY FUNCTIONS FOR CLIENT ---------------- *** # 

def load_client_dataloaders(
        context: Context,
        partition_type,
        num_shards_per_partition,
        batch_size,
        train_test_split_ratio,
        transform=get_transforms
):
    # Retrive meta-data from context
    partition_id = context.node_config["partition-id"]  # assigned at runtime
    num_partitions = context.node_config["num-partitions"]

    # Load flower datasets for clients
    trainloader, valloader = load_flwr_datasets(
        partition_id=partition_id,
        partition_type=partition_type,
        num_partitions=num_partitions,
        num_shards_per_partition=num_shards_per_partition,
        batch_size=batch_size,
        train_test_split_ratio=train_test_split_ratio,
        transform=transform
    )
    return trainloader, valloader

# *** ---------------- CLIENT APP ---------------- *** # 

def get_client_app(
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        strategy=None,
        load_data_fn=load_client_dataloaders,
        batch_size=64,
        partition_type="iid",
        num_shards_per_partition=2,
        train_test_split_ratio=0.2,
        local_epochs=4,
        model_editing=False,
        mask_type='global',
        sparsity=0.2,
        is_save_weights_to_state=False,
        verbose=0,
        mask=None,
        mask_calibration_round=1
) -> ClientApp:
    def client_fn(context: Context):
        print(f"[Client] Client on device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"[Client] CUDA available in client: {torch.cuda.is_available()}")

        trainloader, valloader = load_data_fn(
            context=context,
            partition_type=partition_type,
            batch_size=batch_size,
            num_shards_per_partition=num_shards_per_partition,
            train_test_split_ratio=train_test_split_ratio
        )
        client_state = context.state

        nonlocal strategy # Make strategy defined as param accessible under client_fn
        if strategy == 'standard' or not strategy:
            return CustomNumpyClient(
                client_state=client_state,
                local_epochs=local_epochs,
                trainloader=trainloader,
                valloader=valloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model_editing=model_editing,
                mask_type=mask_type,
                sparsity=sparsity,
                mask_calibration_round=mask_calibration_round,
            ).to_client()
        elif strategy == 'fully_centralized':
            return FullyCentralizedMaskedClient(
                client_state=client_state,
                local_epochs=local_epochs,
                trainloader=trainloader,
                valloader=valloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model_editing=model_editing,
                mask_type=mask_type,
                sparsity=sparsity
            ).to_client()
        elif strategy == 'talos':
            return WarmUpHeadTalosClient(
                client_state=client_state,
                local_epochs=local_epochs,
                trainloader=trainloader,
                valloader=valloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                mask_type=mask_type,
                sparsity=sparsity,
                is_save_weights_to_state=is_save_weights_to_state,
                verbose=verbose,
                mask_calibration_round=mask_calibration_round,
            ).to_client()
        
    app = ClientApp(client_fn=client_fn)
    return app
