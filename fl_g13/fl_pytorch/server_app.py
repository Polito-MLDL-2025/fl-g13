from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from fl_g13.config import RAW_DATA_DIR
from fl_g13.fl_pytorch.datasets import get_eval_transforms
from fl_g13.fl_pytorch.strategy import CustomFedAvg, ClientSideTaskArithmetic
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import load_or_create

# *** -------- UTILITY FUNCTIONS FOR SERVER -------- *** #

def get_evaluate_fn(testloader, model, criterion):
    def evaluate(server_round, parameters_ndarrays, config):
        # Debugging prints
        print(f"[Server Eval Round {server_round}] Model device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
             print(f"[Server Eval Round {server_round}] CUDA available in server eval: {torch.cuda.is_available()}")

        # Applies new parameters to model
        set_weights(model, parameters_ndarrays)

        # Run evaluation and return results
        test_loss, test_accuracy, _ = eval(testloader, model, criterion)
        return test_loss, {"centralized_accuracy": test_accuracy}
    return evaluate

def fit_metrics_aggregation_fn(metrics):
    losses = [n * m["train_loss"] for n, m in metrics]
    drifts = [n * m["drift"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"avg_train_loss": sum(losses) / total, "avg_drift": sum(drifts) / total}

def evaluate_metrics_aggregation_fn(metrics):
    accuracies = [n * m["accuracy"] for n, m in metrics]
    total = sum(n for n, _ in metrics)

    return {"decentralized_avg_eval_accuracy": sum(accuracies) / total}

def simple_scale(client, n_examples, rnd):
    #return 1.0 / n_examples**0.5       # or fixed =1, or metrics based
    return 0.1

# *** -------- SERVER APP -------- *** #

def get_server_app(
    checkpoint_dir,
    prefix,
    model_class,
    model_config=None, ##! New 
    optimizer=None,
    criterion=None,
    scheduler=None,
    device=None,
    save_every=1,
    save_with_model_dir=False,
    #save_best_model=False, ##! Removed
    #get_datatest_fn=get_data_set_default, ##! Removed
    get_evaluate_fn=get_evaluate_fn,
    num_rounds=200,
    fraction_fit=0.1,           # Sample 10% of available clients for training
    fraction_evaluate=0.1,      # Sample 10% of available clients for evaluation
    min_fit_clients=10,         # Never sample less than 10 clients for training
    min_evaluate_clients=10,    # Never sample less than 10 clients for evaluation
    min_available_clients=100,  # Wait until all 100 clients are available
    use_wandb=False,
    wandb_config=None,
    strategy_name="custom_fedavg", ##! New introduced to choose between strategies from notebook
):
    
    # Load or create model if not found in checkpoint_dir
    model, start_epoch = load_or_create(
        path=f"{checkpoint_dir}/{model_class.__name__}" if save_with_model_dir else checkpoint_dir,
        model_class=model_class,
        model_config=model_config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        verbose=True,
    )

    def server_fn(context):
        
        # Debugging prints
        print(f"[Server] Server on device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
             print(f"[Server] CUDA available in client: {torch.cuda.is_available()}")

        # Retrive test dataset and prepare dataloader
        testset = datasets.CIFAR100(RAW_DATA_DIR, train=False, download=True, transform=get_eval_transforms())
        testloader = DataLoader(testset, batch_size=32)
        evaluate_fn = get_evaluate_fn(testloader, model, criterion)

        # Retrive parameters
        params = ndarrays_to_parameters(get_weights(model))

        if strategy_name == "custom_fedavg":
        
            # Call custom strategy for aggregating data
            strategy = CustomFedAvg(
                #run_config=context.run_config, ##! Removed
                checkpoint_dir=checkpoint_dir,
                prefix=prefix,
                model=model,
                start_epoch=start_epoch,
                save_every=save_every,
                save_with_model_dir=save_with_model_dir,
                #save_best_model=save_best_model, ##! Removed
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                initial_parameters=params,
                #on_fit_config_fn=on_fit_config, ##! Removed (on fit config was not used as config was never accessed when needed)
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                use_wandb=use_wandb,
                wandb_config=wandb_config,
            )
        elif strategy_name == "client_side_task_arithmetic":

            strategy = ClientSideTaskArithmetic(
                #run_config=context.run_config, ##! Removed
                checkpoint_dir=checkpoint_dir,
                prefix=prefix,
                model=model,
                start_epoch=start_epoch,
                save_every=save_every,
                save_with_model_dir=save_with_model_dir,
                #save_best_model=save_best_model, ##! Removed
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                initial_parameters=params,
                #on_fit_config_fn=on_fit_config, ##! Removed (on fit config was not used as config was never accessed when needed)
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                use_wandb=use_wandb,
                wandb_config=wandb_config,
                scale_fn=simple_scale,
            )

        # Prepare server config
        rounds = context.run_config.get("num-server-rounds") or num_rounds
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
