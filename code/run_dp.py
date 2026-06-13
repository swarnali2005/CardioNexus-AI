import flwr as fl
from client_dp import HeartClientDP

MU = 0.01

def client_fn(cid: str):
    if cid == "0":
        return HeartClientDP("../data/cleveland_clean.csv", mu=MU).to_client()
    else:
        return HeartClientDP("../data/hungarian_clean.csv", mu=MU).to_client()

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_metrics_average(metrics):
    epsilons = [m["epsilon"] for _, m in metrics]
    noises = [m["noise_multiplier"] for _, m in metrics]
    return {"avg_epsilon": sum(epsilons) / len(epsilons), "avg_noise": sum(noises) / len(noises)}

def fit_config(server_round: int):
    return {"server_round": server_round}

import torch
from model import HeartDiseaseDNN

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = HeartDiseaseDNN(input_dim=10)
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(), "global_model.pt")

        return aggregated_parameters, aggregated_metrics


strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=fit_metrics_average,
    on_fit_config_fn=fit_config,
)
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)

print("\n" + "="*50)
print("FINAL RESULTS - FedProx + Adaptive DP-SGD (mu=0.1)")
print("="*50)
print("Accuracy per round:", history.metrics_distributed["accuracy"])
print("Loss per round:", history.losses_distributed)
print("\nFit metrics per round:", history.metrics_distributed_fit)