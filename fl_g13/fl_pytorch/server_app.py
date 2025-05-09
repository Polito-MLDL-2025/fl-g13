from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader
from torchvision import datasets

from fl_g13.config import RAW_DATA_DIR
from fl_g13.fl_pytorch.datasets import get_eval_transforms
from fl_g13.fl_pytorch.strategy import MaskedFedAvg
from fl_g13.fl_pytorch.task import get_weights, set_weights
from fl_g13.modeling.eval import eval
from fl_g13.modeling.load import load_or_create

# *** -------- UTILITY FUNCTIONS FOR SERVER -------- *** #

def get_evaluate_fn(testloader, model=None, criterion=None):
    def evaluate(server_round, parameters_ndarrays, config):
        set_weights(model, parameters_ndarrays)
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
    return {"federated_evaluate_accuracy": sum(accuracies) / total}

# *** -------- SERVER APP -------- *** #

def get_server_app(
    checkpoint_dir,
    model_class,
    model_config=None,
    optimizer=None,
    criterion=None,
    scheduler=None,
    device=None,
    save_every=1,
    save_best_model=False,
    get_evaluate_fn=get_evaluate_fn,
    num_rounds=200,
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=10,
    min_evaluate_clients=10,
    min_available_clients=100,
    
):
    print(f"Server on {device}")
    
    model, start_epoch = load_or_create(
        path=checkpoint_dir,
        model_class=model_class,
        model_config=model_config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        verbose=True,
    )

    def server_fn(context):

        # Retrive test dataset and prepare dataloader
        testset = datasets.CIFAR100(RAW_DATA_DIR, train=False, download=True, transform=get_eval_transforms())
        testloader = DataLoader(testset, batch_size=32)
        evaluate_fn = get_evaluate_fn(testloader, model, criterion)

        # Retrive parameters
        params = ndarrays_to_parameters(get_weights(model))
        
        # Call custom strategy for aggregating data
        strategy = MaskedFedAvg(
            model=model,
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch,
            save_every=save_every,
            save_best_model=save_best_model,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=params,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        # Prepare server config
        rounds = context.run_config.get("num-server-rounds") or num_rounds
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
