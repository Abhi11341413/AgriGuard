import flwr as fl
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_available_clients=3
)

print("Starting AgriGuard Central Server... Waiting for farmers to connect.")

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
