import flwr as fl
from client import HeartClient

MU = 1.0
def client_fn(cid: str):
    if cid == "0":
        return HeartClient("../data/cleveland_clean.csv", mu=MU).to_client()
    else:
        return HeartClient("../data/hungarian_clean.csv", mu=MU).to_client()

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)

print("\n" + "="*50)
print(f"FINAL RESULTS - FedProx (mu={MU})")
print("="*50)
print("Accuracy per round:", history.metrics_distributed["accuracy"])
print("Loss per round:", history.losses_distributed)